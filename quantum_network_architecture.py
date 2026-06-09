import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import folium
import itertools

class QuantumNetworkBuilder:
    def __init__(self, cities_df, backbone_strategy='delaunay', HUB_THRESHOLD = 500000, country_strategy = 'by_population'):
        self.cities = cities_df
        self.G = nx.Graph()
        self.backbone_strategy = backbone_strategy
        self.country_strategy = country_strategy
        self.hubs_df = cities_df[cities_df['population'] >= HUB_THRESHOLD].copy()
        self.smaller_df = cities_df[(cities_df['population'] < HUB_THRESHOLD) & (cities_df['population'] > 250000)].copy()

    def generate_european_backbone(self):
        """ Layer 1: The Continental Hubs """
        # Add the major cities as nodes
        for idx, row in self.hubs_df.iterrows():
            self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']), type='Hub',
                            pop=row['population'])

        if self.backbone_strategy == "delaunay":
            points = self.hubs_df[['longitude', 'latitude']].values

            # This single line calculates the entire geometric mesh
            tri = Delaunay(points)

            print("3. Building the physical edges in NetworkX...")
            # tri.simplices contains lists of 3 points that make up every triangle in the mesh.
            # We need to extract the 3 edges of each triangle and add them to the graph.
            for simplex in tri.simplices:
                # A simplex looks like [0, 5, 12], which are the row indices in hubs_df
                # We grab the three points of the triangle
                p0, p1, p2 = simplex[0], simplex[1], simplex[2]

                # Define a quick helper function to add an edge between two points
                def add_physical_edge(idx1, idx2):
                    city1 = self.hubs_df.iloc[idx1]['city_name']
                    city2 = self.hubs_df.iloc[idx2]['city_name']

                    # Calculate straight-line Euclidean distance
                    # (Note: For absolute precision in km, you could swap this for a Haversine function)
                    dist = np.linalg.norm(points[idx1] - points[idx2])

                    # NetworkX is smart: if an edge already exists (because triangles share borders),
                    # it will simply ignore the duplicate.
                    self.G.add_edge(city1, city2, weight=dist, type='backbone')

                # Add the 3 edges of the triangle
                add_physical_edge(p0, p1)
                add_physical_edge(p1, p2)
                add_physical_edge(p2, p0)

        elif self.backbone_strategy == "mst":

            G_complete = nx.Graph()
            G_complete.add_nodes_from(self.G.nodes(data=True))  # Copy your hubs over

            print("Calculating every possible fiber route...")
            # itertools.combinations gets every unique pair of cities (e.g., Paris-Berlin, Paris-Madrid)
            for u, v in itertools.combinations(G_complete.nodes(), 2):
                pos_u = np.array(G_complete.nodes[u]['pos'])
                pos_v = np.array(G_complete.nodes[v]['pos'])

                # Calculate physical distance
                dist = np.linalg.norm(pos_u - pos_v)

                # Add the hypothetical edge
                G_complete.add_edge(u, v, weight=dist)

            # 2. Carve away the fat using the MST algorithm!
            print("Carving the Minimum Spanning Tree...")
            mst_backbone = nx.minimum_spanning_tree(G_complete, weight='weight')

            # Mark these edges as our backbone
            for u, v, data in mst_backbone.edges(data=True):
                mst_backbone[u][v]['type'] = 'backbone'

            self.G = mst_backbone.copy()

        elif self.backbone_strategy == "k-nearest-neighbor":
            knn_backbone = nx.Graph()
            knn_backbone.add_nodes_from(self.G.nodes(data=True))

            K_NEIGHBORS = 2  # Change to 3 if you want a denser, more redundant mesh

            print(f"Connecting every hub to its {K_NEIGHBORS} closest neighbors...")

            for u in knn_backbone.nodes():
                pos_u = np.array(knn_backbone.nodes[u]['pos'])

                # Create a list to store the distances to every other city
                distances = []

                for v in knn_backbone.nodes():
                    if u != v:  # Don't measure the distance to itself!
                        pos_v = np.array(knn_backbone.nodes[v]['pos'])
                        dist = np.linalg.norm(pos_u - pos_v)
                        distances.append((dist, v))

                # Sort the list so the shortest distances are at the beginning
                distances.sort()

                # Grab the top K closest neighbors and draw the edges!
                for i in range(K_NEIGHBORS):
                    dist_nearest = distances[i][0]
                    v_nearest = distances[i][1]

                    # NetworkX will naturally ignore this if the reverse edge (v to u) was already added
                    knn_backbone.add_edge(u, v_nearest, weight=dist_nearest, type='backbone')

            self.G = knn_backbone.copy()
        elif self.backbone_strategy == "satellite":
            # Add a 'Satellite' node and connect top 5 hubs to it directly
            pass
        elif self.backbone_strategy == "hybrid":
            G_complete = nx.Graph()
            G_complete.add_nodes_from(self.G.nodes(data=True))

            for u, v in itertools.combinations(G_complete.nodes(), 2):
                dist = np.linalg.norm(np.array(G_complete.nodes[u]['pos']) - np.array(G_complete.nodes[v]['pos']))
                G_complete.add_edge(u, v, weight=dist)

            # --- STEP 1: The Lifeline (MST) ---
            # This guarantees 100% global connectivity with zero islands.
            mst_graph = nx.minimum_spanning_tree(G_complete, weight='weight')

            for u, v in mst_graph.edges():
                mst_graph[u][v]['type'] = 'backbone_mst'  # Tag it so we know it's a critical lifeline

            # --- STEP 2: The Local Redundancy (k-NN) ---
            # This adds the smart, short local overlapping rings.
            knn_graph = nx.Graph()
            knn_graph.add_nodes_from(self.G.nodes(data=True))
            K_NEIGHBORS = 4

            for u in knn_graph.nodes():
                pos_u = np.array(knn_graph.nodes[u]['pos'])
                distances = []

                for v in knn_graph.nodes():
                    if u != v:
                        dist = np.linalg.norm(pos_u - np.array(knn_graph.nodes[v]['pos']))
                        distances.append((dist, v))

                distances.sort()
                for i in range(K_NEIGHBORS):
                    dist_nearest = distances[i][0]
                    v_nearest = distances[i][1]
                    knn_graph.add_edge(u, v_nearest, weight=dist_nearest, type='backbone_knn')
            self.G = nx.compose(mst_graph, knn_graph)

        else:
            raise ValueError("backbone_strategy must be 'delaunay', 'mst', 'satellite'")

        return self.G

    def generate_country(self):

        if self.country_strategy == 'by_population':
            """
            Connects real cities by chaining them to the nearest active node.
            """

            # Sort by population descending so larger secondary cities seed first
            smaller_sorted = self.smaller_df.sort_values(by='population', ascending=False)

            for idx, row in smaller_sorted.iterrows():
                new_city = row['city_name']
                new_pos = np.array([row['longitude'], row['latitude']])

                # Search the graph for the closest existing node
                min_dist = float('inf')
                target_node = None

                for n, data in self.G.nodes(data=True):
                    existing_pos = np.array(data['pos'])
                    dist = np.linalg.norm(new_pos - existing_pos)

                    if dist < min_dist:
                        min_dist = dist
                        target_node = n

                # Add the new city and connect it to its closest neighbor
                self.G.add_node(new_city, pos=tuple(new_pos), pop=row['population'], type='Smaller')
                self.G.add_edge(new_city, target_node, weight=min_dist, type='access')

        elif self.country_strategy == 'ba':
            """
                Generates local complex networks scaled by the Hub's population.
                """
            print("Building Alt 2: Hierarchical Fractal Sub-Networks...")
            scaling_factor = 100000
            # Isolate the Hubs currently in the graph
            hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Hub']

            for hub in hubs:
                hub_pop = self.G.nodes[hub]['pop']
                hub_pos = self.G.nodes[hub]['pos']

                # Calculate how many local nodes to generate based on population
                # (e.g., 3,000,000 pop / 100,000 = 30 nodes)
                N_local = max(1, int(hub_pop / scaling_factor))

                # Generate a local complex network (m=1 makes it highly tree-like)
                local_G = nx.barabasi_albert_graph(N_local, 1)

                # Relabel nodes so they don't overwrite nodes from other cities
                mapping = {i: f"{hub}_user_{i}" for i in range(N_local)}
                nx.relabel_nodes(local_G, mapping, copy=False)

                # Give them spatial coordinates scattered closely around the Hub
                for n in local_G.nodes():
                    # Add a microscopic random offset (approx 5-15 km radius)
                    offset_lon = np.random.normal(0, 0.05)
                    offset_lat = np.random.normal(0, 0.05)

                    local_G.nodes[n]['pos'] = (hub_pos[0] + offset_lon, hub_pos[1] + offset_lat)
                    local_G.nodes[n]['type'] = 'End_User'

                # Merge the local cluster into the main European backbone
                self.G = nx.compose(self.G, local_G)

                # Link the local cluster to the Hub (connecting the local "Node 0")
                self.G.add_edge(hub, f"{hub}_user_0", weight=0.01, type='access')

        elif self.country_strategy == 'hybrid':
            alpha = 0.5
            beta = 0.5
            """
                Attaches real cities using a weighted probability of Degree and Population.
                """
            print("Building Alt 3: Hybrid Pop + Degree Preferential Attachment...")

            for idx, row in self.smaller_df.iterrows():
                new_city = row['city_name']
                new_pos = np.array([row['longitude'], row['latitude']])

                nodes = list(self.G.nodes())
                scores = []

                # Calculate the gravity score for every node currently in the graph
                for n in nodes:
                    k_i = self.G.degree(n)  # The topological degree
                    pop_i = self.G.nodes[n].get('pop', 10000)  # The population size

                    # The core mathematical formula you designed
                    score = (alpha * k_i) + (beta * pop_i)
                    scores.append(score)

                # Convert raw scores into an array of probabilities (must sum to 1.0)
                total_score = sum(scores)
                probabilities = [s / total_score for s in scores]

                # Use numpy to randomly pick the target node based on those exact probabilities
                target_node = np.random.choice(nodes, p=probabilities)

                # Calculate the actual physical distance for the edge weight
                target_pos = np.array(self.G.nodes[target_node]['pos'])
                dist = np.linalg.norm(new_pos - target_pos)

                # Add the node and the edge
                self.G.add_node(new_city, pos=tuple(new_pos), pop=row['population'], type='Spoke')
                self.G.add_edge(new_city, target_node, weight=dist, type='access')

        return self.G

    def plot_network(self, G):
        m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles="CartoDB positron")

        # 2. Draw the Edges (Quantum Channels) FIRST so they stay in the background
        print("Drawing network edges...")
        for u, v, edge_data in G.edges(data=True):
            # Get the coordinates from the nodes
            u_lon, u_lat = G.nodes[u]['pos']
            v_lon, v_lat = G.nodes[v]['pos']

            # Check the edge type to color-code the lines
            if edge_data.get('type') == 'backbone':
                line_color = "#00FFFF"  # Cyan for the main backbone
                line_weight = 2
                line_opacity = 0.8
            else:
                line_color = "#808080"  # Grey for regional access links
                line_weight = 1
                line_opacity = 0.4

            # Draw the line (Remember: Folium needs [Lat, Lon]!)
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=line_color,
                weight=line_weight,
                opacity=line_opacity
            ).add_to(m)

        # 3. Draw the Nodes (Cities) SECOND so they sit on top of the lines
        print("Drawing network nodes...")
        for node_name, node_data in G.nodes(data=True):
            lon, lat = node_data['pos']
            pop = node_data.get('pop', 0)

            # Style based on whether it's a Hub or a Spoke
            if node_data.get('type') == 'Hub':
                node_color = "#FF3366"  # Bright pink/red for Hubs
                node_radius = 3
            else:
                node_color = "#FF3366"  # White for smaller access nodes
                node_radius = 0.5

            # Add the interactive bubble
            folium.CircleMarker(
                location=[lat, lon],
                radius=node_radius,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.9,
                popup=f"<b>{node_name}</b><br>Type: {node_data.get('type')}<br>Pop: {int(pop):,}"
            ).add_to(m)

        # 4. Save and view!
        m.save("quantum_network_map.html")
        print("Map successfully saved as 'quantum_network_map.html'!")

df = pd.read_csv('final_city_coordinates.csv')
arch = QuantumNetworkBuilder(df, backbone_strategy='hybrid', country_strategy='hybrid')
G = arch.generate_european_backbone()
G = arch.generate_country()
arch.plot_network(G)