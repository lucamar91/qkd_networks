import pandas as pd
import numpy as np
import networkx as nx
import scipy
from matplotlib.scale import scale_factory
from scipy.spatial import Delaunay, KDTree
import folium
import itertools
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
from folium import Element
from global_land_mask import globe

class QuantumNetworkBuildertest:
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
#arch = QuantumNetworkBuilder(df, backbone_strategy='hybrid', country_strategy='by_population')
#G = arch.generate_european_backbone()
#G = arch.generate_country()
#arch.plot_network(G)

class QuantumNetworkBuilder:
    def __init__(self, cities_df, d_max=100):
        self.cities = cities_df
        self.G = nx.Graph()
        self.d_max = d_max
        self.tier1_df = cities_df[cities_df['population'] >= 1000000].copy()
        self.tier2_df = cities_df[(cities_df['population'] < 1000000) & (cities_df['population'] > 100000)].copy()
        self.tier3_df = cities_df[cities_df['population'] <= 100000].copy()

    def generate_tier1(self, max_distance=400):
        """
        Generates the Tier 1 backbone, dynamically spawning Relay hubs in dead zones,
        and connecting them using a sparse Waxman probability model.
        """
        print("Generating Tier 1 Backbone...")

        # ---------------------------------------------------------
        # 1. ADD ORIGINAL TIER 1 HUBS (RED)
        # ---------------------------------------------------------
        hubs_coords_list = []
        for idx, row in self.tier1_df.iterrows():
            self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                            type='Tier1_Hub', pop=row['population'], color='#FF0000')  # Red
            # Store in radians for fast math later
            hubs_coords_list.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        # Convert to a dynamic numpy array that we can add to
        active_hubs_rad = np.array(hubs_coords_list)

        # ---------------------------------------------------------
        # 2. THE COVERAGE STEP: GREEDY RELAY SPAWNING (PURPLE)
        # ---------------------------------------------------------
        # Sort descending! This guarantees the BIGGEST city in a dead zone becomes the Relay.
        smaller_cities = pd.concat([self.tier2_df, self.tier3_df]).sort_values(by='population', ascending=False)

        for idx, row in smaller_cities.iterrows():
            # Get coordinates of this specific city in radians
            city_rad = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])

            # Fast numpy calculation to all CURRENTLY ACTIVE hubs
            dists = haversine_distances(city_rad, active_hubs_rad) * 6371

            # If this city is further than max_distance from the closest active hub...
            if np.min(dists) > max_distance:
                # Promote it!
                self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                                type='Tier1_Relay', pop=row['population'], color='#9400D3')  # Purple

                # Add it to the active hubs list. This "protects" nearby small towns
                # from also promoting themselves in subsequent loops!
                active_hubs_rad = np.vstack([active_hubs_rad, city_rad])

        # ---------------------------------------------------------
        # 3. VECTORIZED EDGE GENERATION (MST + k-NN)
        # ---------------------------------------------------------
        node_names = list(self.G.nodes())

        # Extract coordinates of all final hubs
        final_coords_deg = np.array([[self.G.nodes[n]['pos'][1], self.G.nodes[n]['pos'][0]] for n in node_names])
        final_coords_rad = np.radians(final_coords_deg)

        # Generate the full distance matrix instantly
        dist_matrix = haversine_distances(final_coords_rad, final_coords_rad) * 6371

        # --- Part A: The Lifeline (Minimum Spanning Tree) ---
        # We build a temporary graph of ALL possible connections to let NetworkX carve the MST
        G_temp = nx.Graph()
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])

        mst = nx.minimum_spanning_tree(G_temp, weight='weight')

        # Add the MST edges to our main graph (Tag them White)
        for u, v, data in mst.edges(data=True):
            self.G.add_edge(u, v, weight=data['weight'], edge_type='T1_MST', color='#FFFFFF')

        # --- Part B: Local Mesh Redundancy (k-Nearest Neighbors) ---
        K_NEIGHBORS = 3

        for i in range(len(node_names)):
            # np.argsort sorts the distances from smallest to largest.
            # Index 0 is the city's distance to itself (0 km).
            # Indices 1 through K are the closest neighbors!
            nearest_indices = np.argsort(dist_matrix[i])[1: K_NEIGHBORS + 1]

            for j in nearest_indices:
                u_name = node_names[i]
                v_name = node_names[j]
                distance = dist_matrix[i, j]

                # NetworkX automatically ignores duplicates if the MST already added this edge.
                # If it's a new edge, we add it and tag it Cyan!
                if not self.G.has_edge(u_name, v_name):
                    self.G.add_edge(u_name, v_name, weight=distance, edge_type='T1_KNN', color='#00FFFF')

        print(f"Tier 1 complete! Total Hubs: {self.G.number_of_nodes()}, Total Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier2(self, max_peer_distance=150, t2_capacity=200000):
        """
        Attaches real mid-sized cities to Tier 1, AND fractures massive Tier 1 hubs
        into synthetic Tier 2 district switches based on population.
        """
        print("Generating Tier 2 Aggregators (Real & Synthetic)...")

        # --- PART A: ADD REAL TIER 2 CITIES ---
        t2_names = []
        t2_coords_rad = []

        for idx, row in self.tier2_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier2_Aggregator', pop=row['population'], color='#FFA500')  # Orange
            t2_names.append(name)
            t2_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        t2_coords_rad = np.array(t2_coords_rad)

        # --- PART B: EXTRACT ACTIVE TIER 1 HUBS ---
        t1_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') in ['Tier1_Hub', 'Tier1_Relay']]
        t1_coords_rad = np.array([
            [np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])]
            for n in t1_nodes
        ])

        # --- PART C: CONNECT REAL T2 TO NEAREST T1 ---
        if len(t2_coords_rad) > 0 and len(t1_coords_rad) > 0:
            t1_tree = BallTree(t1_coords_rad, metric='haversine')
            distances, indices = t1_tree.query(t2_coords_rad, k=1)

            for i, t2_name in enumerate(t2_names):
                t1_idx = indices[i][0]
                t1_name = t1_nodes[t1_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t2_name, t1_name, weight=dist_km, edge_type='T2_to_T1', color='#FFA500')

        # --- PART D: FRACTAL SPAWNING (SYNTHETIC TIER 2s) ---
        for t1_name in t1_nodes:
            pop = self.G.nodes[t1_name].get('pop', 0)

            # If the hub is massive, calculate how many T2 district switches it needs
            if pop > t2_capacity:
                num_synthetic = int(pop / t2_capacity)
                base_lon, base_lat = self.G.nodes[t1_name]['pos']

                for i in range(num_synthetic):
                    syn_name = f"{t1_name}_syn_T2_{i}"

                    # Jitter by ~5-8 km to keep them inside the metropolitan area
                    syn_lon = base_lon + np.random.normal(0, 0.05)
                    syn_lat = base_lat + np.random.normal(0, 0.05)

                    # We divide the population evenly among the new synthetic switches
                    # so Tier 3 knows exactly how to fracture them later!
                    syn_pop = pop / num_synthetic

                    self.G.add_node(syn_name, pos=(syn_lon, syn_lat),
                                    type='Tier2_Synthetic', pop=syn_pop, color='#FFDAB9')  # Light Orange

                    # Assume a standard 5km metropolitan fiber run to the main hub
                    self.G.add_edge(syn_name, t1_name, weight=5.0, edge_type='T2_Synthetic_Link', color='#FFDAB9')

        print(f"Tier 2 complete! Total Nodes: {self.G.number_of_nodes()}, Total Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier3(self, t3_capacity=25000):
        """
        Attaches real small towns to Tier 2, AND fractures all Tier 2 nodes
        (both real and synthetic) into local neighborhood switches.
        """
        print("Generating Tier 3 Switches (Real & Synthetic)...")

        # --- PART A: ADD REAL TIER 3 CITIES ---
        t3_names = []
        t3_coords_rad = []

        for idx, row in self.tier3_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier3_Switch', pop=row['population'], color='#00FF00')  # Green
            t3_names.append(name)
            t3_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        t3_coords_rad = np.array(t3_coords_rad)

        # --- PART B: EXTRACT ACTIVE TIER 2 HUBS (Real & Synthetic!) ---
        t2_nodes = [n for n, attr in self.G.nodes(data=True) if
                    attr.get('type') in ['Tier2_Aggregator', 'Tier2_Synthetic']]
        t2_coords_rad = np.array([
            [np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])]
            for n in t2_nodes
        ])

        # --- PART C: CONNECT REAL T3 TO NEAREST T2 ---
        if len(t3_coords_rad) > 0 and len(t2_coords_rad) > 0:
            t2_tree = BallTree(t2_coords_rad, metric='haversine')
            distances, indices = t2_tree.query(t3_coords_rad, k=1)

            for i, t3_name in enumerate(t3_names):
                t2_idx = indices[i][0]
                t2_name = t2_nodes[t2_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t3_name, t2_name, weight=dist_km, edge_type='T3_to_T2', color='#00FF00')

        # --- PART D: FRACTAL SPAWNING (SYNTHETIC TIER 3s) ---
        for t2_name in t2_nodes:
            pop = self.G.nodes[t2_name].get('pop', 0)

            if pop > t3_capacity:
                num_synthetic = int(pop / t3_capacity)
                base_lon, base_lat = self.G.nodes[t2_name]['pos']

                for i in range(num_synthetic):
                    syn_name = f"{t2_name}_syn_T3_{i}"

                    # Jitter by ~2-3 km (Neighborhood level)
                    syn_lon = base_lon + np.random.normal(0, 0.02)
                    syn_lat = base_lat + np.random.normal(0, 0.02)

                    syn_pop = pop / num_synthetic

                    self.G.add_node(syn_name, pos=(syn_lon, syn_lat),
                                    type='Tier3_Synthetic', pop=syn_pop, color='#98FB98')  # Light Green

                    # Assume a standard 2km local fiber run
                    self.G.add_edge(syn_name, t2_name, weight=2.0, edge_type='T3_Synthetic_Link', color='#98FB98')

        print(f"Tier 3 complete! Total Nodes: {self.G.number_of_nodes()}, Total Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier4(self, scale_factor=10000):
        """
        Generates Tier 4 End Users by dynamically spawning them around both
        Real and Synthetic Tier 3 switches based on local population capacity.
        """
        print("Generating Tier 4 End Users...")

        # 1. THE FIX: Accept both real Eurostat towns and our synthetic city neighborhoods!
        valid_t3_types = ['Tier3_Switch', 'Tier3_Synthetic']
        t3_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') in valid_t3_types]

        for n in t3_nodes:
            # Calculate N users based on the fractured population
            pop = self.G.nodes[n].get('pop', 0)
            N = max(1, int(pop / scale_factor))

            # Grab the coordinates of the parent switch
            base_lon, base_lat = self.G.nodes[n]['pos']

            for i in range(N):
                user_id = f"{n}_user_{i}"

                # Microscopic coordinate jitter (approx 1 - 2 km radius)
                user_lon = base_lon + np.random.normal(0, 0.01)
                user_lat = base_lat + np.random.normal(0, 0.01)

                self.G.add_node(user_id,
                                pos=(user_lon, user_lat),
                                type='Tier4_User',
                                color='#0000FF')  # Blue

                self.G.add_edge(user_id, n, weight=1.5, edge_type='T4_Access', color='#0000FF')

        print(f"Tier 4 complete! Total Nodes: {self.G.number_of_nodes()}, Total Links: {self.G.number_of_edges()}")
        return self.G
    def add_repeaters(self):
        pass

    def plot_network(self):
        """
        Renders the complex network onto an interactive Folium map.
        Nodes and edges are styled dynamically based on their tags.
        """
        print("Generating Folium Map...")

        # Initialize Map (Dark theme makes coloured lines pop)
        m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles="CartoDB dark_matter", prefer_canvas=True)

        # ---------------------------------------------------------
        # 1. DRAW EDGES FIRST (So they render underneath the nodes)
        # ---------------------------------------------------------
        for u, v, edge_data in self.G.edges(data=True):
            # Retrieve coordinates
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']

            # Read the edge tags (Fallback to grey if tag is missing)
            e_color = edge_data.get('color', '#808080')
            e_type = edge_data.get('edge_type', 'Unknown')

            # Make backbone thicker, access links thinner
            e_weight = 2.5 if e_type == 'T1_Backbone' else 1.0
            e_opacity = 0.8 if e_type == 'T1_Backbone' else 0.4

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],  # Folium needs Lat, Lon!
                color=e_color,
                weight=e_weight,
                opacity=e_opacity,
                popup=f"Type: {e_type}"
            ).add_to(m)

        # ---------------------------------------------------------
        # 2. DRAW NODES SECOND (So they sit on top)
        # ---------------------------------------------------------
        for node_id, node_data in self.G.nodes(data=True):
            lon, lat = node_data['pos']

            # Read the node tags
            n_color = node_data.get('color', '#FFFFFF')
            n_type = node_data.get('type', 'Unknown')

            # Size nodes based on their tier
            if n_type == 'Tier1_Hub':
                n_radius = 5
            elif n_type == 'Tier2_Aggregator':
                n_radius = 3
            elif n_type == 'Repeater':
                n_radius = 1
            else:
                n_radius = 2

            folium.CircleMarker(
                location=[lat, lon],
                radius=n_radius,
                color=n_color,
                fill=True,
                fill_color=n_color,
                fill_opacity=0.9,
                popup=f"<b>{node_id}</b><br>Tier: {n_type}"
            ).add_to(m)

        # ---------------------------------------------------------
        # 3. ADD THE SCREENSHOT BUTTON (JavaScript Injection)
        # ---------------------------------------------------------
        # Import the html2canvas library so the browser can convert HTML to an image
        html2canvas_src = '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>'
        m.get_root().html.add_child(Element(html2canvas_src))

        # Create the visual button and the Javascript logic
        screenshot_js = """
        <div style="position: absolute; top: 10px; left: 50px; z-index: 9999;">
            <button id="capture-btn" style="padding: 10px; background-color: white; border: 2px solid rgba(0,0,0,0.2); border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif;">
                📸 Take Screenshot
            </button>
        </div>

        <script>
            document.getElementById('capture-btn').addEventListener('click', function() {
                // Ask the user for a filename, default to a unique timestamped name
                var defaultName = "quantum_network_" + Date.now();
                var filename = prompt("Enter a name for your screenshot (without .png):", defaultName);

                if (filename) {
                    // Temporarily hide the button so it doesn't show up in the photo
                    var btn = document.getElementById('capture-btn');
                    btn.style.display = 'none';

                    // Grab the Leaflet map container
                    var mapContainer = document.querySelector('.leaflet-container');

                    // Convert the map to an image
                    html2canvas(mapContainer, {
                        useCORS: true, // This is critical! It allows the script to download the map background tiles
                        allowTaint: false
                    }).then(function(canvas) {
                        // Create a fake link, attach the image, and click it to download
                        var link = document.createElement('a');
                        link.download = filename + '.png';
                        link.href = canvas.toDataURL('image/png');
                        link.click();

                        // Make the button visible again
                        btn.style.display = 'block';
                    });
                }
            });
        </script>
        """
        m.get_root().html.add_child(Element(screenshot_js))
        m.save("quantum_network_map.html")

#net = QuantumNetworkBuilder(df)


import networkx as nx
import pandas as pd
import numpy as np
import folium
import random
from folium import Element
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
from global_land_mask import globe


class QuantumNetworkBuilder_App1:
    def __init__(self, cities_df, d_max=100):
        self.cities = cities_df
        self.G = nx.Graph()
        self.d_max = d_max
        self.tier1_df = cities_df[cities_df['population'] >= 1000000].copy()
        self.tier2_df = cities_df[(cities_df['population'] < 1000000) & (cities_df['population'] > 100000)].copy()
        self.tier3_df = cities_df[cities_df['population'] <= 100000].copy()

    def generate_tier1(self, max_distance=400):
        print("Generating Tier 1 Backbone...")
        hubs_coords_list = []
        for idx, row in self.tier1_df.iterrows():
            self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                            type='Tier1_Hub', pop=row['population'], color='#FF0000')
            hubs_coords_list.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        active_hubs_rad = np.array(hubs_coords_list)
        smaller_cities = pd.concat([self.tier2_df, self.tier3_df]).sort_values(by='population', ascending=False)

        for idx, row in smaller_cities.iterrows():
            city_rad = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
            dists = haversine_distances(city_rad, active_hubs_rad) * 6371
            if np.min(dists) > max_distance:
                self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                                type='Tier1_Relay', pop=row['population'], color='#9400D3')
                active_hubs_rad = np.vstack([active_hubs_rad, city_rad])

        node_names = list(self.G.nodes())
        final_coords_deg = np.array([[self.G.nodes[n]['pos'][1], self.G.nodes[n]['pos'][0]] for n in node_names])
        final_coords_rad = np.radians(final_coords_deg)
        dist_matrix = haversine_distances(final_coords_rad, final_coords_rad) * 6371

        G_temp = nx.Graph()
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])

        mst = nx.minimum_spanning_tree(G_temp, weight='weight')
        for u, v, data in mst.edges(data=True):
            self.G.add_edge(u, v, weight=data['weight'], edge_type='T1_MST', color='#FFFFFF')

        K_NEIGHBORS = 3
        for i in range(len(node_names)):
            nearest_indices = np.argsort(dist_matrix[i])[1: K_NEIGHBORS + 1]
            for j in nearest_indices:
                u_name = node_names[i]
                v_name = node_names[j]
                distance = dist_matrix[i, j]
                if not self.G.has_edge(u_name, v_name):
                    self.G.add_edge(u_name, v_name, weight=distance, edge_type='T1_KNN', color='#00FFFF')

        print(f"Tier 1 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier2(self):
        print("Generating Tier 2 Aggregators (Real Cities Only)...")
        t2_names = []
        t2_coords_rad = []
        for idx, row in self.tier2_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier2_Aggregator', pop=row['population'], color='#FFA500')
            t2_names.append(name)
            t2_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])
        t2_coords_rad = np.array(t2_coords_rad)

        t1_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') in ['Tier1_Hub', 'Tier1_Relay']]
        t1_coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in t1_nodes])

        if len(t2_coords_rad) > 0 and len(t1_coords_rad) > 0:
            t1_tree = BallTree(t1_coords_rad, metric='haversine')
            distances, indices = t1_tree.query(t2_coords_rad, k=1)
            for i, t2_name in enumerate(t2_names):
                t1_idx = indices[i][0]
                t1_name = t1_nodes[t1_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t2_name, t1_name, weight=dist_km, edge_type='T2_to_T1', color='#FFA500')

        print(f"Tier 2 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier3(self):
        print("Generating Tier 3 Switches (Real Cities Only)...")
        t3_names = []
        t3_coords_rad = []
        for idx, row in self.tier3_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier3_Switch', pop=row['population'], color='#00FF00')
            t3_names.append(name)
            t3_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])
        t3_coords_rad = np.array(t3_coords_rad)

        t2_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier2_Aggregator']
        t2_coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in t2_nodes])

        if len(t3_coords_rad) > 0 and len(t2_coords_rad) > 0:
            t2_tree = BallTree(t2_coords_rad, metric='haversine')
            distances, indices = t2_tree.query(t3_coords_rad, k=1)
            for i, t3_name in enumerate(t3_names):
                t2_idx = indices[i][0]
                t2_name = t2_nodes[t2_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t3_name, t2_name, weight=dist_km, edge_type='T3_to_T2', color='#00FF00')

        print(f"Tier 3 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier4(self, total_switches=2000, total_users=15000):
        print("Generating Tier 4: Voronoi + Dual Barabási Overlay...")

        # 1. Pull coordinates for all anchor nodes (Tier 1, 2, 3)
        anchor_nodes = list(self.G.nodes())
        coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in anchor_nodes])

        # Build the tree to instantly find Voronoi territories
        territory_tree = BallTree(coords_rad, metric='haversine')

        # Keep track of which nodes belong to which cluster
        cluster_members = {anchor: [anchor] for anchor in anchor_nodes}

        # Bounding Box for Europe
        lat_min, lat_max = 35.0, 70.0
        lon_min, lon_max = -10.0, 30.0

        # ---------------------------------------------------------
        # PHASE 1: SPAWN SYNTHETIC SWITCHES (m = 3)
        # ---------------------------------------------------------
        print(f"Phase 1: Spawning {total_switches} Synthetic Switches (Mesh Layer)...")
        switches_added = 0

        while switches_added < total_switches:
            test_lat = random.uniform(lat_min, lat_max)
            test_lon = random.uniform(lon_min, lon_max)

            if globe.is_land(test_lat, test_lon):
                switch_id = f"Voronoi_Switch_{switches_added}"
                self.G.add_node(switch_id, pos=(test_lon, test_lat), type='Tier3_Synthetic',
                                color='#98FB98')  # Light Green

                # Whose territory are we in?
                new_pt_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])
                _, indices = territory_tree.query(new_pt_rad, k=1)
                owner_anchor = anchor_nodes[indices[0][0]]
                current_cluster = cluster_members[owner_anchor]

                # BARABÁSI STEP (m=3)
                # We connect to 3 nodes, unless the cluster is still too small, then we connect to whatever is available
                m = min(3, len(current_cluster))

                degrees = [self.G.degree(c) for c in current_cluster]
                total_degree = sum(degrees)
                probs = [deg / total_degree for deg in degrees]

                # Pick 'm' unique targets based on degree probability
                target_nodes = np.random.choice(current_cluster, size=m, replace=False, p=probs)

                for target_node in target_nodes:
                    t_lon, t_lat = self.G.nodes[target_node]['pos']
                    d_km = haversine_distances(new_pt_rad, [[np.radians(t_lat), np.radians(t_lon)]])[0][0] * 6371

                    self.G.add_edge(switch_id, target_node, weight=d_km, edge_type='T3_Synthetic_Link', color='#98FB98')

                # Add this switch to the cluster so users can connect to it!
                cluster_members[owner_anchor].append(switch_id)
                switches_added += 1

        # ---------------------------------------------------------
        # PHASE 2: SPAWN END USERS (m = 1)
        # ---------------------------------------------------------
        print(f"Phase 2: Spawning {total_users} End Users (Access Layer)...")
        users_added = 0

        while users_added < total_users:
            test_lat = random.uniform(lat_min, lat_max)
            test_lon = random.uniform(lon_min, lon_max)

            if globe.is_land(test_lat, test_lon):
                user_id = f"Voronoi_User_{users_added}"
                self.G.add_node(user_id, pos=(test_lon, test_lat), type='Tier4_User', color='#0000FF')  # Blue

                # Whose territory are we in?
                new_pt_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])
                _, indices = territory_tree.query(new_pt_rad, k=1)
                owner_anchor = anchor_nodes[indices[0][0]]
                current_cluster = cluster_members[owner_anchor]

                # BARABÁSI STEP (m=1)
                degrees = [self.G.degree(c) for c in current_cluster]
                total_degree = sum(degrees)
                probs = [deg / total_degree for deg in degrees]

                target_node = np.random.choice(current_cluster, size=1, p=probs)[0]

                t_lon, t_lat = self.G.nodes[target_node]['pos']
                d_km = haversine_distances(new_pt_rad, [[np.radians(t_lat), np.radians(t_lon)]])[0][0] * 6371

                self.G.add_edge(user_id, target_node, weight=d_km, edge_type='T4_Access', color='#0000FF')

                cluster_members[owner_anchor].append(user_id)
                users_added += 1

        print(f"Tier 4 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier5(self, scale_factor=20000, min_users_per_node=3):
        print("Generating Tier 4: True Localized Voronoi Complex Networks...")

        # 1. Pull coordinates for all anchor nodes (Tier 1, 2, 3)
        anchor_nodes = list(self.G.nodes())
        coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in anchor_nodes])

        # Build the tree to define Voronoi territories and find nearest neighbors
        territory_tree = BallTree(coords_rad, metric='haversine')
        distances, _ = territory_tree.query(coords_rad, k=2)

        total_users_added = 0

        # Iterate over EVERY SINGLE NODE to build its own local complex network
        for i, anchor in enumerate(anchor_nodes):
            # Calculate how many users this specific territory gets based on population
            pop = self.G.nodes[anchor].get('pop', 5000)  # Default to 5000 for rural relays
            N_users = max(min_users_per_node, int(pop / scale_factor))

            base_lon, base_lat = self.G.nodes[anchor]['pos']

            # Find distance to nearest neighbor.
            # We multiply by 1.5 to ensure we search far enough to hit the corners of the Voronoi cell!
            search_radius_km = max(10.0, (distances[i][1] * 6371) * 1.5)

            # Start the cluster with the anchor hub
            current_cluster = [anchor]

            users_added_here = 0
            attempts = 0

            # Rejection Sampling: Keep trying until we fill this node's quota
            while users_added_here < N_users and attempts < 1000:
                attempts += 1

                # Generate a random point within the search radius
                r = random.uniform(0, search_radius_km)
                theta = random.uniform(0, 2 * np.pi)

                dlat = (r * np.sin(theta)) / 111.0
                dlon = (r * np.cos(theta)) / (111.0 * np.cos(np.radians(base_lat)))

                test_lat = base_lat + dlat
                test_lon = base_lon + dlon

                # CHECK 1: Is it on land?
                if globe.is_land(test_lat, test_lon):

                    # CHECK 2: Is it ACTUALLY inside this anchor's Voronoi cell?
                    new_pt_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])
                    _, indices = territory_tree.query(new_pt_rad, k=1)
                    owner_anchor = anchor_nodes[indices[0][0]]

                    if owner_anchor == anchor:
                        # SUCCESS! This point is legally inside our territory.
                        user_id = f"{anchor}_User_{users_added_here}"
                        self.G.add_node(user_id, pos=(test_lon, test_lat), type='Tier4_User', color='#0000FF')

                        # LOCAL BARABÁSI (m=1): Grow the network outward
                        degrees = [self.G.degree(c) for c in current_cluster]
                        total_degree = sum(degrees)
                        probs = [deg / total_degree for deg in degrees]

                        target_node = np.random.choice(current_cluster, p=probs)

                        t_lon, t_lat = self.G.nodes[target_node]['pos']
                        d_km = haversine_distances(new_pt_rad, [[np.radians(t_lat), np.radians(t_lon)]])[0][0] * 6371

                        self.G.add_edge(user_id, target_node, weight=d_km, edge_type='T4_Access', color='#0000FF')

                        # Add to the cluster so the next node can attach to this one
                        current_cluster.append(user_id)
                        users_added_here += 1
                        total_users_added += 1

        print(f"Tier 4 complete! Added {total_users_added} total users.")
        print(f"Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def add_repeaters(self):
        pass

    def plot_network(self):
        print("Generating Folium Map...")
        m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles="CartoDB dark_matter", prefer_canvas=True)

        for u, v, edge_data in self.G.edges(data=True):
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']
            e_color = edge_data.get('color', '#808080')
            e_type = edge_data.get('edge_type', 'Unknown')
            e_weight = 2.5 if e_type == 'T1_Backbone' else 1.0
            e_opacity = 0.8 if e_type == 'T1_Backbone' else 0.4

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=e_color, weight=e_weight, opacity=e_opacity, popup=f"Type: {e_type}"
            ).add_to(m)

        for node_id, node_data in self.G.nodes(data=True):
            lon, lat = node_data['pos']
            n_color = node_data.get('color', '#FFFFFF')
            n_type = node_data.get('type', 'Unknown')
            n_radius = 5 if n_type == 'Tier1_Hub' else 3 if n_type == 'Tier2_Aggregator' else 1 if n_type == 'Repeater' else 2

            folium.CircleMarker(
                location=[lat, lon], radius=n_radius, color=n_color, fill=True,
                fill_color=n_color, fill_opacity=0.9, popup=f"<b>{node_id}</b><br>Tier: {n_type}"
            ).add_to(m)

        html2canvas_src = '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>'
        m.get_root().html.add_child(Element(html2canvas_src))
        screenshot_js = """
        <div style="position: absolute; top: 10px; left: 50px; z-index: 9999;">
            <button id="capture-btn" style="padding: 10px; background-color: white; border: 2px solid rgba(0,0,0,0.2); border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif;">
                📸 Take Screenshot
            </button>
        </div>
        <script>
            window.addEventListener('load', function() {
                var btn = document.getElementById('capture-btn');
                if (btn) {
                    btn.addEventListener('click', function() {
                        var defaultName = "quantum_network_" + Date.now();
                        var filename = prompt("Enter a name for your screenshot (without .png):", defaultName);
                        if (filename) {
                            btn.style.display = 'none';
                            var mapContainer = document.querySelector('.leaflet-container');
                            html2canvas(mapContainer, {useCORS: true, allowTaint: false}).then(function(canvas) {
                                var link = document.createElement('a');
                                link.download = filename + '.png';
                                link.href = canvas.toDataURL('image/png');
                                link.click();
                                btn.style.display = 'block';
                            });
                        }
                    });
                }
            });
        </script>
        """
        m.get_root().html.add_child(Element(screenshot_js))
        m.save("quantum_network_app1.html")

#df = pd.read_csv('final_city_coordinates.csv')
net = QuantumNetworkBuilder_App1(df)


import networkx as nx
import pandas as pd
import numpy as np
import folium
import random
from folium import Element
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
from global_land_mask import globe


class QuantumNetworkBuilder_App2:
    def __init__(self, cities_df, d_max=100):
        self.cities = cities_df
        self.G = nx.Graph()
        self.d_max = d_max
        self.tier1_df = cities_df[cities_df['population'] >= 1000000].copy()
        self.tier2_df = cities_df[(cities_df['population'] < 1000000) & (cities_df['population'] > 100000)].copy()
        self.tier3_df = cities_df[cities_df['population'] <= 100000].copy()

    def generate_tier1(self, max_distance=400):
        print("Generating Tier 1 Backbone...")
        hubs_coords_list = []
        for idx, row in self.tier1_df.iterrows():
            self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                            type='Tier1_Hub', pop=row['population'], color='#FF0000')
            hubs_coords_list.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        active_hubs_rad = np.array(hubs_coords_list)
        smaller_cities = pd.concat([self.tier2_df, self.tier3_df]).sort_values(by='population', ascending=False)

        for idx, row in smaller_cities.iterrows():
            city_rad = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
            dists = haversine_distances(city_rad, active_hubs_rad) * 6371
            if np.min(dists) > max_distance:
                self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                                type='Tier1_Relay', pop=row['population'], color='#9400D3')
                active_hubs_rad = np.vstack([active_hubs_rad, city_rad])

        node_names = list(self.G.nodes())
        final_coords_deg = np.array([[self.G.nodes[n]['pos'][1], self.G.nodes[n]['pos'][0]] for n in node_names])
        final_coords_rad = np.radians(final_coords_deg)
        dist_matrix = haversine_distances(final_coords_rad, final_coords_rad) * 6371

        G_temp = nx.Graph()
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])

        mst = nx.minimum_spanning_tree(G_temp, weight='weight')
        for u, v, data in mst.edges(data=True):
            self.G.add_edge(u, v, weight=data['weight'], edge_type='T1_MST', color='#FFFFFF')

        K_NEIGHBORS = 3
        for i in range(len(node_names)):
            nearest_indices = np.argsort(dist_matrix[i])[1: K_NEIGHBORS + 1]
            for j in nearest_indices:
                u_name = node_names[i]
                v_name = node_names[j]
                distance = dist_matrix[i, j]
                if not self.G.has_edge(u_name, v_name):
                    self.G.add_edge(u_name, v_name, weight=distance, edge_type='T1_KNN', color='#00FFFF')

        print(f"Tier 1 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier2(self):
        print("Generating Tier 2 Aggregators (Real Cities Only)...")
        t2_names = []
        t2_coords_rad = []
        for idx, row in self.tier2_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier2_Aggregator', pop=row['population'], color='#FFA500')
            t2_names.append(name)
            t2_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])
        t2_coords_rad = np.array(t2_coords_rad)

        t1_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') in ['Tier1_Hub', 'Tier1_Relay']]
        t1_coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in t1_nodes])

        if len(t2_coords_rad) > 0 and len(t1_coords_rad) > 0:
            t1_tree = BallTree(t1_coords_rad, metric='haversine')
            distances, indices = t1_tree.query(t2_coords_rad, k=1)
            for i, t2_name in enumerate(t2_names):
                t1_idx = indices[i][0]
                t1_name = t1_nodes[t1_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t2_name, t1_name, weight=dist_km, edge_type='T2_to_T1', color='#FFA500')

        print(f"Tier 2 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier3(self):
        print("Generating Tier 3 Switches (Real Cities Only)...")
        t3_names = []
        t3_coords_rad = []
        for idx, row in self.tier3_df.iterrows():
            name = row['city_name']
            self.G.add_node(name, pos=(row['longitude'], row['latitude']),
                            type='Tier3_Switch', pop=row['population'], color='#00FF00')
            t3_names.append(name)
            t3_coords_rad.append([np.radians(row['latitude']), np.radians(row['longitude'])])
        t3_coords_rad = np.array(t3_coords_rad)

        t2_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier2_Aggregator']
        t2_coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in t2_nodes])

        if len(t3_coords_rad) > 0 and len(t2_coords_rad) > 0:
            t2_tree = BallTree(t2_coords_rad, metric='haversine')
            distances, indices = t2_tree.query(t3_coords_rad, k=1)
            for i, t3_name in enumerate(t3_names):
                t2_idx = indices[i][0]
                t2_name = t2_nodes[t2_idx]
                dist_km = distances[i][0] * 6371
                self.G.add_edge(t3_name, t2_name, weight=dist_km, edge_type='T3_to_T2', color='#00FF00')

        print(f"Tier 3 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def generate_tier4(self, total_switches=5000, total_users=15000):
        print("Generating Tier 4: Spatially-Embedded Dual BA Overlay...")

        # Pull coordinates for all active nodes to serve as the structural backbone
        active_nodes = list(self.G.nodes())
        active_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in active_nodes])

        # Bounding Box for Europe roughly (Lat: 35 to 70, Lon: -10 to 30)
        lat_min, lat_max = 35.0, 70.0
        lon_min, lon_max = -10.0, 30.0

        # ---------------------------------------------------------
        # PHASE 1: SPAWN SYNTHETIC SWITCHES (m = 3)
        # ---------------------------------------------------------
        print(f"Phase 1: Spawning {total_switches} Synthetic Switches (Mesh Layer)...")
        switches_added = 0

        while switches_added < total_switches:
            test_lat = random.uniform(lat_min, lat_max)
            test_lon = random.uniform(lon_min, lon_max)

            # Check if random coordinate is on actual land
            if globe.is_land(test_lat, test_lon):
                switch_id = f"BA_Europe_Switch_{switches_added}"
                self.G.add_node(switch_id, pos=(test_lon, test_lat), type='Tier3_Synthetic', color='#98FB98')

                # Math: Find distances to all active nodes
                new_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])
                dists_km = haversine_distances(new_rad, active_coords)[0] * 6371

                # Performance trick: Evaluate the 15 closest nodes for preferential attachment
                n_closest = min(15, len(active_nodes))
                closest_indices = np.argpartition(dists_km, n_closest - 1)[:n_closest]

                weights = []
                for idx in closest_indices:
                    target_name = active_nodes[idx]
                    k = self.G.degree(target_name)
                    d = dists_km[idx]

                    # Spatially-Embedded BA Formula: P = Degree / (Distance^2)
                    w = k / ((d + 1) ** 2)
                    weights.append(w)

                # Normalize probabilities
                total_w = sum(weights)
                probs = [w / total_w for w in weights]

                # Roll the dice to pick m=3 unique targets
                m_edges = min(3, len(closest_indices))
                chosen_indices = np.random.choice(closest_indices, size=m_edges, replace=False, p=probs)

                # Draw the mesh edges
                for chosen_idx in chosen_indices:
                    target_node = active_nodes[chosen_idx]
                    final_dist = dists_km[chosen_idx]
                    self.G.add_edge(switch_id, target_node, weight=final_dist, edge_type='T3_Synthetic_Link',
                                    color='#98FB98')

                # Add the new switch into the active arrays so future nodes can attach to it!
                active_nodes.append(switch_id)
                active_coords = np.vstack([active_coords, new_rad])

                switches_added += 1

        # ---------------------------------------------------------
        # PHASE 2: SPAWN END USERS (m = 1)
        # ---------------------------------------------------------
        print(f"Phase 2: Spawning {total_users} End Users (Access Layer)...")
        users_added = 0

        while users_added < total_users:
            test_lat = random.uniform(lat_min, lat_max)
            test_lon = random.uniform(lon_min, lon_max)

            if globe.is_land(test_lat, test_lon):
                user_id = f"BA_Europe_User_{users_added}"
                self.G.add_node(user_id, pos=(test_lon, test_lat), type='Tier4_User', color='#0000FF')

                new_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])
                dists_km = haversine_distances(new_rad, active_coords)[0] * 6371

                # Evaluate the 10 closest nodes (End-users don't need to look as far)
                n_closest = min(10, len(active_nodes))
                closest_indices = np.argpartition(dists_km, n_closest - 1)[:n_closest]

                weights = []
                for idx in closest_indices:
                    target_name = active_nodes[idx]
                    k = self.G.degree(target_name)
                    d = dists_km[idx]
                    w = k / ((d + 1) ** 2)
                    weights.append(w)

                total_w = sum(weights)
                probs = [w / total_w for w in weights]

                # Roll the dice to pick exactly m=1 target
                chosen_idx = np.random.choice(closest_indices, p=probs)

                target_node = active_nodes[chosen_idx]
                final_dist = dists_km[chosen_idx]

                self.G.add_edge(user_id, target_node, weight=final_dist, edge_type='T4_Access', color='#0000FF')

                # Add the user to the active arrays
                active_nodes.append(user_id)
                active_coords = np.vstack([active_coords, new_rad])

                users_added += 1

        print(f"Tier 4 complete! Total Nodes: {self.G.number_of_nodes()}, Links: {self.G.number_of_edges()}")
        return self.G

    def add_repeaters(self):
        pass

    def plot_network(self):
        print("Generating Folium Map...")
        m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles="CartoDB dark_matter", prefer_canvas=True)

        for u, v, edge_data in self.G.edges(data=True):
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']
            e_color = edge_data.get('color', '#808080')
            e_type = edge_data.get('edge_type', 'Unknown')
            e_weight = 2.5 if e_type == 'T1_Backbone' else 1.0
            e_opacity = 0.8 if e_type == 'T1_Backbone' else 0.4

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=e_color, weight=e_weight, opacity=e_opacity, popup=f"Type: {e_type}"
            ).add_to(m)

        for node_id, node_data in self.G.nodes(data=True):
            lon, lat = node_data['pos']
            n_color = node_data.get('color', '#FFFFFF')
            n_type = node_data.get('type', 'Unknown')
            n_radius = 5 if n_type == 'Tier1_Hub' else 3 if n_type == 'Tier2_Aggregator' else 1 if n_type == 'Repeater' else 2

            folium.CircleMarker(
                location=[lat, lon], radius=n_radius, color=n_color, fill=True,
                fill_color=n_color, fill_opacity=0.9, popup=f"<b>{node_id}</b><br>Tier: {n_type}"
            ).add_to(m)

        html2canvas_src = '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>'
        m.get_root().html.add_child(Element(html2canvas_src))
        screenshot_js = """
        <div style="position: absolute; top: 10px; left: 50px; z-index: 9999;">
            <button id="capture-btn" style="padding: 10px; background-color: white; border: 2px solid rgba(0,0,0,0.2); border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif;">
                📸 Take Screenshot
            </button>
        </div>
        <script>
            window.addEventListener('load', function() {
                var btn = document.getElementById('capture-btn');
                if (btn) {
                    btn.addEventListener('click', function() {
                        var defaultName = "quantum_network_" + Date.now();
                        var filename = prompt("Enter a name for your screenshot (without .png):", defaultName);
                        if (filename) {
                            btn.style.display = 'none';
                            var mapContainer = document.querySelector('.leaflet-container');
                            html2canvas(mapContainer, {useCORS: true, allowTaint: false}).then(function(canvas) {
                                var link = document.createElement('a');
                                link.download = filename + '.png';
                                link.href = canvas.toDataURL('image/png');
                                link.click();
                                btn.style.display = 'block';
                            });
                        }
                    });
                }
            });
        </script>
        """
        m.get_root().html.add_child(Element(screenshot_js))
        m.save("quantum_network_app2.html")

# df = pd.read_csv('final_city_coordinates.csv')
#net = QuantumNetworkBuilder_App2(df)
G = net.generate_tier1()
G = net.generate_tier2()
G = net.generate_tier3()
G = net.generate_tier4()
G =net.generate_tier5()
net.plot_network()