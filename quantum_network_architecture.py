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
import os
import requests
import pandas as pd
import zipfile
import io
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union

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

    def generate_tier4(self, total_nodes=3000, target_countries=['Spain', 'United Kingdom']):
        print(f"Generating Tier 4: Regional Overlap + Border Stitching (Filtered for {target_countries})...")

        # 1. Map every original city to its Administrative Region
        city_to_region = dict(zip(self.cities['city_name'], self.cities['region']))

        # 2. Extract active anchors (T1, T2, T3) and build the territory tree
        anchor_nodes = list(self.G.nodes())
        anchor_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in anchor_nodes])
        territory_tree = BallTree(anchor_coords, metric='haversine')

        # 3. Create the Regional Rosters (Dictionary holding all nodes per region)
        regional_rosters = {}
        allowed_anchors = set()

        for anchor in anchor_nodes:
            matching_rows = self.cities[self.cities['city_name'] == anchor]
            if not matching_rows.empty:
                country = matching_rows.iloc[0]['country']
                region = matching_rows.iloc[0]['region']

                # Apply the country filter for debugging
                if country in target_countries:
                    allowed_anchors.add(anchor)
                    if region not in regional_rosters:
                        regional_rosters[region] = []
                    regional_rosters[region].append(anchor)

        # 4. Pre-calculate target ratios (15% Enterprise m=3, 85% Standard m=1)
        n_enterprise = int(total_nodes * 0.15)

        # Active lists to track coordinates as the network grows
        active_nodes = list(self.G.nodes())
        active_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in active_nodes])

        lat_min, lat_max = 35.0, 70.0
        lon_min, lon_max = -10.0, 30.0

        nodes_added = 0

        while nodes_added < total_nodes:
            test_lat = random.uniform(lat_min, lat_max)
            test_lon = random.uniform(lon_min, lon_max)

            if globe.is_land(test_lat, test_lon):
                new_rad = np.array([[np.radians(test_lat), np.radians(test_lon)]])

                # Find the TWO closest anchors to check for borders!
                dists, indices = territory_tree.query(new_rad, k=2)

                anchor_1 = anchor_nodes[indices[0][0]]
                if anchor_1 not in allowed_anchors:
                    continue  # Skip if dart landed outside our target countries

                region_1 = city_to_region.get(anchor_1, 'Unknown')

                # IDEA A: THE BORDER PROXIMITY CHECK
                allowed_regions = [region_1]

                # If the 2nd closest city belongs to a different region AND is within 30km...
                if len(indices[0]) > 1:
                    anchor_2 = anchor_nodes[indices[0][1]]
                    region_2 = city_to_region.get(anchor_2, 'Unknown')
                    dist_2_km = dists[0][1] * 6371

                    if region_1 != region_2 and dist_2_km < 30.0 and region_2 in regional_rosters:
                        # DUAL CITIZENSHIP GRANTED! This node will stitch the border.
                        allowed_regions.append(region_2)

                # Combine the rosters of the allowed regions
                available_targets = []
                for r in allowed_regions:
                    if r in regional_rosters:
                        available_targets.extend(regional_rosters[r])

                if len(available_targets) == 0:
                    continue

                # Determine if this is an Enterprise (m=3) or Standard (m=1) User
                if nodes_added < n_enterprise:
                    node_id = f"Enterprise_User_{nodes_added}"
                    n_type = 'Tier4_Enterprise'
                    n_color = '#00FFFF'  # Cyan for high-capacity users
                    m = min(3, len(available_targets))
                else:
                    node_id = f"Standard_User_{nodes_added}"
                    n_type = 'Tier4_User'
                    n_color = '#0000FF'  # Blue for standard users
                    m = 1

                self.G.add_node(node_id, pos=(test_lon, test_lat), type=n_type, color=n_color)

                # Fetch coordinates only for the allowed regional targets to save CPU
                target_coords = np.array(
                    [[np.radians(self.G.nodes[t]['pos'][1]), np.radians(self.G.nodes[t]['pos'][0])] for t in
                     available_targets])
                dists_km = haversine_distances(new_rad, target_coords)[0] * 6371

                # To mimic scale-free growth, pick the 15 closest nodes WITHIN the region to evaluate
                n_closest = min(15, len(available_targets))
                closest_indices = np.argpartition(dists_km, n_closest - 1)[:n_closest]

                weights = []
                for idx in closest_indices:
                    target_name = available_targets[idx]
                    k = self.G.degree(target_name)
                    d = dists_km[idx]

                    # SOFTENED DISTANCE PENALTY: 1 / (d + 1) instead of squared!
                    # This lets the complex network breathe and fill the dead space.
                    w = k / (d + 1)
                    weights.append(w)

                total_w = sum(weights)
                probs = [w / total_w for w in weights]

                # Roll the dice to pick 'm' unique targets
                chosen_indices = np.random.choice(closest_indices, size=m, replace=False, p=probs)

                for chosen_idx in chosen_indices:
                    target_node = available_targets[chosen_idx]
                    final_dist = dists_km[chosen_idx]
                    self.G.add_edge(node_id, target_node, weight=final_dist, edge_type='T4_Access', color=n_color)

                # Add the new node into the active lists AND the regional roster
                # so future nodes can attach to this one!
                active_nodes.append(node_id)
                active_coords = np.vstack([active_coords, new_rad])
                regional_rosters[region_1].append(node_id)

                nodes_added += 1

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
        m.save("quantum_network_app1.html")

#df = pd.read_csv('final_city_coordinates.csv')
#net = QuantumNetworkBuilder_App1(df)


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


import pandas as pd
import numpy as np
import networkx as nx
import random
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances
import folium
from branca.element import Element


class QuantumNetworkBuilder_App3:
    def __init__(self, cities_df, d_max=100):
        self.cities = cities_df
        self.G = nx.Graph()
        self.d_max = d_max

        # Categorize the real cities
        self.tier1_df = cities_df[cities_df['population'] >= 1000000].copy()
        self.tier2_df = cities_df[(cities_df['population'] < 1000000) & (cities_df['population'] > 100000)].copy()
        self.tier3_df = cities_df[cities_df['population'] <= 100000].copy()

    def generate_tier1(self, max_distance=250):
        print("Tier 1: Generating Continental Core Backbone (All of Europe)...")
        hubs_coords_list = []
        for idx, row in self.tier1_df.iterrows():
            self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                            type='Tier1_Hub', pop=row['population'], region=row.get('region', 'Unknown'),
                            color='#FF0000')
            hubs_coords_list.append([np.radians(row['latitude']), np.radians(row['longitude'])])

        active_hubs_rad = np.array(hubs_coords_list)
        smaller_cities = pd.concat([self.tier2_df, self.tier3_df]).sort_values(by='population', ascending=False)

        for idx, row in smaller_cities.iterrows():
            city_rad = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
            dists = haversine_distances(city_rad, active_hubs_rad) * 6371
            if np.min(dists) > max_distance:
                self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                                type='Tier1_Relay', pop=row['population'], region=row.get('region', 'Unknown'),
                                color='#9400D3')
                active_hubs_rad = np.vstack([active_hubs_rad, city_rad])

        # Draw the MST + KNN Mesh
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

        for i in range(len(node_names)):
            nearest_indices = np.argsort(dist_matrix[i])[1: 4]  # K=3
            for j in nearest_indices:
                u, v = node_names[i], node_names[j]
                if not self.G.has_edge(u, v):
                    self.G.add_edge(u, v, weight=dist_matrix[i, j], edge_type='T1_KNN', color='#00FFFF')

        print(f"Tier 1 Complete. Nodes: {self.G.number_of_nodes()}")
        return self.G

    def _attach_spatially_embedded(self, new_node, new_rad, m_edges, active_nodes, active_coords, edge_type,
                                   edge_color):
        """Helper function to perform Spatially-Embedded BA attachment"""
        dists_km = haversine_distances(new_rad, active_coords)[0] * 6371
        n_closest = min(15, len(active_nodes))
        closest_indices = np.argpartition(dists_km, n_closest - 1)[:n_closest]

        weights = []
        for idx in closest_indices:
            k = self.G.degree(active_nodes[idx])
            d = dists_km[idx]
            # SSFN Formula: Degree / Distance
            weights.append(k / (d + 1))

        probs = [w / sum(weights) for w in weights]
        m = min(m_edges, len(closest_indices))
        chosen_indices = np.random.choice(closest_indices, size=m, replace=False, p=probs)

        for idx in chosen_indices:
            self.G.add_edge(new_node, active_nodes[idx], weight=dists_km[idx], edge_type=edge_type, color=edge_color)

    def generate_tier2(self, target_countries=['Spain', 'United Kingdom'], gap_km=80, min_pop_geonames=50000):
        print(f"Tier 2: Generating Density-Controlled National Backhaul ({target_countries})...")

        country_code_map = {
            'Spain': 'ES', 'United Kingdom': 'GB', 'France': 'FR',
            'Germany': 'DE', 'Italy': 'IT', 'Portugal': 'PT',
        }

        # --- STEP 1: Base candidates from Eurostat (unchanged logic) ---
        candidates = pd.concat([
            self.tier2_df[self.tier2_df['country'].isin(target_countries)],
            self.tier3_df[self.tier3_df['country'].isin(target_countries)]
        ]).sort_values(by='population', ascending=False)
        candidates = candidates[~candidates['city_name'].isin(self.G.nodes())]

        # --- STEP 2: Load GeoNames as a supplement, sorted by population ---
        geonames_supplements = []
        for country in target_countries:
            cc = country_code_map.get(country)
            if not cc:
                continue
            gdf = self.get_regional_data(country_code=cc)
            gdf['country'] = country
            gdf = gdf.rename(columns={'name': 'city_name'})
            geonames_supplements.append(gdf)

        geonames_df = pd.concat(geonames_supplements).sort_values(by='population', ascending=False)
        # Only consider GeoNames cities above the population threshold
        geonames_df = geonames_df[geonames_df['population'] >= min_pop_geonames]

        # Build a BallTree over Eurostat candidate positions to detect gaps
        eurostat_coords_rad = np.array([
            [np.radians(row['latitude']), np.radians(row['longitude'])]
            for _, row in candidates.iterrows()
        ])
        eurostat_tree = BallTree(eurostat_coords_rad, metric='haversine')

        # Find GeoNames cities that are far from ANY Eurostat city (i.e. fill a real gap)
        gap_fill_rows = []
        for _, row in geonames_df.iterrows():
            if row['city_name'] in candidates['city_name'].values:
                continue  # Already in Eurostat, skip
            pt = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
            dist_rad, _ = eurostat_tree.query(pt, k=1)
            dist_km = dist_rad[0][0] * 6371
            if dist_km > gap_km:
                gap_fill_rows.append(row)

        if gap_fill_rows:
            gap_fill_df = pd.DataFrame(gap_fill_rows)
            # Deduplicate: if two GeoNames cities fill the same gap, keep highest population
            # Do this by clustering them against each other
            used_indices = set()
            deduped = []
            gap_coords_rad = np.array([
                [np.radians(r['latitude']), np.radians(r['longitude'])]
                for r in gap_fill_rows
            ])
            gap_tree = BallTree(gap_coords_rad, metric='haversine')

            for i, row in enumerate(gap_fill_rows):
                if i in used_indices:
                    continue
                pt = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
                # Mark all GeoNames cities within gap_km of this one as "consumed"
                indices = gap_tree.query_radius(pt, r=gap_km / 6371)
                for idx in indices[0]:
                    if idx != i:
                        used_indices.add(idx)
                deduped.append(row)  # This is already the highest-pop one (df is sorted)

            gap_fill_df = pd.DataFrame(deduped)
            candidates = pd.concat([candidates, gap_fill_df]).sort_values(by='population', ascending=False)

        # --- STEP 3: Everything below here is identical to your original logic ---
        candidates = candidates[~candidates['city_name'].isin(self.G.nodes())]

        active_t1_nodes = [n for n, attr in self.G.nodes(data=True) if 'Tier1' in attr.get('type', '')]
        active_t1_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in active_t1_nodes])

        active_t2_nodes = []
        active_t2_coords = []

        all_active_nodes = list(self.G.nodes())
        all_active_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in all_active_nodes])

        for idx, row in candidates.iterrows():
            name = row['city_name']
            pop = row['population']
            new_rad = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])

            dists_to_all = haversine_distances(new_rad, all_active_coords)[0] * 6371
            min_dist = np.min(dists_to_all)

            if pop > 100000 or min_dist > gap_km:
                self.G.add_node(name, pos=(row['longitude'], row['latitude']), type='Tier2_Aggregator',
                                pop=pop, region=row.get('region', row.get('admin1_code', 'Unknown')), color='#FFA500')

                if len(active_t1_nodes) > 0:
                    dists_to_t1 = haversine_distances(new_rad, active_t1_coords)[0] * 6371
                    nearest_t1_idx = np.argmin(dists_to_t1)
                    t1_target = active_t1_nodes[nearest_t1_idx]
                    self.G.add_edge(name, t1_target, weight=dists_to_t1[nearest_t1_idx],
                                    edge_type='T2_to_T1', color='#FFA500')

                if len(active_t2_nodes) > 0:
                    t2_coords_arr = np.vstack(active_t2_coords)
                    dists_to_t2 = haversine_distances(new_rad, t2_coords_arr)[0] * 6371
                    valid_indices = np.where(dists_to_t2 < 200.0)[0]

                    if len(valid_indices) > 0:
                        weights = []
                        for v_idx in valid_indices:
                            k = self.G.degree(active_t2_nodes[v_idx])
                            d = dists_to_t2[v_idx]
                            weights.append((k + 1) / (d + 1))
                        total_w = sum(weights)
                        probs = [w / total_w for w in weights]
                        m = min(1, len(valid_indices))
                        chosen = np.random.choice(valid_indices, size=m, replace=False, p=probs)
                        for c_idx in chosen:
                            self.G.add_edge(name, active_t2_nodes[c_idx], weight=dists_to_t2[c_idx],
                                            edge_type='T2_to_T2', color='#FFA500')

                active_t2_nodes.append(name)
                active_t2_coords = new_rad if len(active_t2_coords) == 0 else np.vstack([active_t2_coords, new_rad])
                all_active_nodes.append(name)
                all_active_coords = np.vstack([all_active_coords, new_rad])

        print(f"Tier 2 Complete. Added {len(active_t2_nodes)} Regional Nodes.")
        return self.G

    def generate_tier4_fractal(self, target_regions=['Catalonia'], m1=2, m2=1, p=0.15, pop_scale=1000):
        print(f"Executing Integrated Tier 4 Fractal Access Layer for {target_regions}...")

        # 1. Identify Anchors (T1, T2 & T3 hubs)
        hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('region') in target_regions and (
                'Tier1' in attr.get('type', '') or
                'Tier2' in attr.get('type', '') or
                'Tier3' in attr.get('type', ''))]

        if not hubs:
            print("No backbone anchors found.")
            return self.G

        hub_coords_dict = {n: np.array([[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])]])
                           for n in hubs}

        # --- COMPLETELY DYNAMIC BORDER PATROL SETUP ---
        print(f"Mobilizing dynamic international border patrol for {target_regions[0]}...")

        # 1. Dynamically calculate the bounding box based on the known region
        known_region_cities = self.cities[self.cities['region'] == target_regions[0]]
        if known_region_cities.empty:
            print(f"Error: No known cities in {target_regions[0]} to establish borders.")
            return self.G

        min_lat, max_lat = known_region_cities['latitude'].min(), known_region_cities['latitude'].max()
        min_lon, max_lon = known_region_cities['longitude'].min(), known_region_cities['longitude'].max()

        # Add a 1.0 degree buffer (~111km) to catch international border spillovers
        patrol_bounds = (min_lat - 1.0, max_lat + 1.0, min_lon - 1.0, max_lon + 1.0)

        # 2. Detect which countries overlap this buffer zone using your base CSV
        buffer_cities = self.cities[
            (self.cities['latitude'] >= patrol_bounds[0]) & (self.cities['latitude'] <= patrol_bounds[1]) &
            (self.cities['longitude'] >= patrol_bounds[2]) & (self.cities['longitude'] <= patrol_bounds[3])
            ]

        # Standard Eurostat to GeoNames ISO mapping
        iso_map = {
            'Spain': 'ES', 'France': 'FR', 'Andorra': 'AD', 'Portugal': 'PT',
            'Germany': 'DE', 'Italy': 'IT', 'United Kingdom': 'GB', 'Belgium': 'BE',
            'Netherlands': 'NL', 'Switzerland': 'CH', 'Austria': 'AT', 'Poland': 'PL'
        }

        # Extract unique country codes found in the buffer
        target_isos = []
        if 'country' in buffer_cities.columns:
            neighboring_countries = buffer_cities['country'].unique()
            target_isos = [iso_map.get(c) for c in neighboring_countries if c in iso_map]

        # Ensure the home country is always in the list
        home_country = known_region_cities.iloc[0]['country'] if 'country' in known_region_cities.columns else 'Spain'
        home_iso = iso_map.get(home_country, 'ES')
        if home_iso not in target_isos:
            target_isos.append(home_iso)

        # Edge-case: Eurostat often skips microstates. If we detect a Spain/France border, manually inject Andorra.
        if 'ES' in target_isos and 'FR' in target_isos and 'AD' not in target_isos:
            target_isos.append('AD')

        # 3. Load Customs Agents for all detected surrounding countries
        patrol_dfs = []
        for iso in set(target_isos):
            try:
                df_country = self.get_regional_data(country_code=iso, region_bounds=patrol_bounds)
                patrol_dfs.append(df_country)
            except Exception as e:
                pass  # Gracefully skip if a country download fails

        full_patrol_df = pd.concat(patrol_dfs).reset_index(drop=True)

        # 4. Find the target region's true admin code (using ONLY the home country to avoid border-town mistakes)
        anchor_lat = known_region_cities.iloc[0]['latitude']
        anchor_lon = known_region_cities.iloc[0]['longitude']

        home_df = full_patrol_df[full_patrol_df['country_code'] == home_iso].reset_index(drop=True)
        distances = (home_df['latitude'] - anchor_lat) ** 2 + (home_df['longitude'] - anchor_lon) ** 2
        target_admin_code = home_df.loc[distances.idxmin(), 'admin1_code']

        # 5. Build the International Border Tree
        country_coords = np.array(
            [[np.radians(row['latitude']), np.radians(row['longitude'])] for _, row in full_patrol_df.iterrows()])
        country_admin_codes = full_patrol_df['admin1_code'].values
        country_iso_codes = full_patrol_df['country_code'].values

        border_tree = BallTree(country_coords, metric='haversine')
        # ----------------------------------------------

        all_new_t4_nodes = []
        users_added = 0

        # 2. Per-Hub Fractal Spawning
        for hub in hubs:
            hub_lon, hub_lat = self.G.nodes[hub]['pos']
            pop = self.G.nodes[hub].get('pop', 50000)
            num_users = min(400, max(20, int(pop / pop_scale)))

            dists = [haversine_distances(hub_coords_dict[hub], coords)[0][0] * 6371 for other_hub, coords in
                     hub_coords_dict.items() if hub != other_hub]
            safe_radius_km = np.max(dists) #if dists else 25.0
            dynamic_scale_deg = max(0.05, (num_users / 400.0) * (safe_radius_km / 111.0))

            H = nx.Graph()
            H.add_node(hub)
            active_nodes = [hub]
            local_new_nodes = []

            for i in range(num_users):
                user_id = f"{hub}_User_{i}"
                H.add_node(user_id)
                local_new_nodes.append(user_id)
                all_new_t4_nodes.append(user_id)

                m = m1 if random.random() < p else m2
                degrees = np.array([H.degree(t) for t in active_nodes])
                probs = degrees / degrees.sum() if degrees.sum() > 0 else np.ones(len(active_nodes)) / len(active_nodes)

                chosen = np.random.choice(active_nodes, size=min(m, len(active_nodes)), replace=False, p=probs)
                for c in chosen:
                    H.add_edge(user_id, c, edge_type='T4_Access')
                active_nodes.append(user_id)

            abstract_pos = nx.spring_layout(H, scale=dynamic_scale_deg, iterations=40)
            hub_abs_lon, hub_abs_lat = abstract_pos[hub]

            for n in local_new_nodes:
                px = abstract_pos[n][0] + (hub_lon - hub_abs_lon)
                py = abstract_pos[n][1] + (hub_lat - hub_abs_lat)

                # 1. Ocean Rotation: Keep it on land
                angle = 0
                while not globe.is_land(py, px) and angle < 360:
                    angle += 15
                    theta = np.radians(angle)
                    rel_x, rel_y = px - hub_lon, py - hub_lat
                    px = hub_lon + (np.cos(theta) * rel_x) - (np.sin(theta) * rel_y)
                    py = hub_lat + (np.sin(theta) * rel_x) + (np.cos(theta) * rel_y)

                # 2. STRICT ADMIN BORDER PATROL: Keep it inside the target region
                attempts = 0
                while attempts < 20:
                    pt = np.array([[np.radians(py), np.radians(px)]])
                    _, idx = border_tree.query(pt, k=1)

                    nearest_admin = country_admin_codes[idx[0][0]]
                    nearest_country = country_iso_codes[idx[0][0]]

                    # DYNAMIC CHECK: Matches target region AND target country
                    if nearest_admin == target_admin_code and nearest_country == home_iso:
                        break  # Safe!

                    # It spilled over a border! Pull it 15% closer to the hub and check again
                    px = hub_lon + (px - hub_lon) * 0.85
                    py = hub_lat + (py - hub_lat) * 0.85
                    attempts += 1

                self.G.add_node(n, pos=(px, py), type='Tier4_User', region=target_regions[0])
                for edge in H.edges(n):
                    self.G.add_edge(edge[0], edge[1], edge_type='T4_Access')
            users_added += num_users

        # 3. Topological Weaving (Fuzzy Borders)
        regional_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get('region') in target_regions]
        regional_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in regional_nodes])
        tree = BallTree(regional_coords, metric='haversine')

        for n in all_new_t4_nodes:
            if self.G.degree(n) < 2: continue

            n_rad = np.array([[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])]])
            indices = tree.query_radius(n_rad, r=12.0 / 6371)[0]

            valid_targets = [t for t in [regional_nodes[i] for i in indices] if
                             t != n and not self.G.has_edge(n, t) and n.split('_')[0] != t.split('_')[0]]

            if valid_targets:
                degrees = np.array([self.G.degree(t) for t in valid_targets])
                probs = (degrees + 1) / (degrees + 1).sum()
                self.G.add_edge(n, np.random.choice(valid_targets, p=probs), edge_type='T4_Fuzzy_Link')

        print(f"Tier 4 Complete. Added {users_added} users securely inside borders.")
        return self.G

    def generate_tier4_unified(self, target_regions=['Catalonia'], m1=2, m2=1, p=0.15, pop_scale=1000):
        print(f"Executing Unified Tier 4 Macro-Fractal for {target_regions}...")

        # 1. Identify Regional Backbone Anchors
        hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('region') in target_regions and (
                'Tier1' in attr.get('type', '') or
                'Tier2' in attr.get('type', '') or
                'Tier3' in attr.get('type', ''))]

        if not hubs:
            print("No backbone anchors found.")
            return self.G

        # --- COMPLETELY DYNAMIC BORDER PATROL SETUP ---
        print(f"Mobilizing dynamic international border patrol for {target_regions[0]}...")

        known_region_cities = self.cities[self.cities['region'] == target_regions[0]]
        min_lat, max_lat = known_region_cities['latitude'].min(), known_region_cities['latitude'].max()
        min_lon, max_lon = known_region_cities['longitude'].min(), known_region_cities['longitude'].max()
        patrol_bounds = (min_lat - 1.0, max_lat + 1.0, min_lon - 1.0, max_lon + 1.0)

        buffer_cities = self.cities[
            (self.cities['latitude'] >= patrol_bounds[0]) & (self.cities['latitude'] <= patrol_bounds[1]) &
            (self.cities['longitude'] >= patrol_bounds[2]) & (self.cities['longitude'] <= patrol_bounds[3])
            ]

        iso_map = {'Spain': 'ES', 'France': 'FR', 'Andorra': 'AD', 'Portugal': 'PT', 'Germany': 'DE', 'Italy': 'IT'}
        target_isos = [iso_map.get(c) for c in buffer_cities['country'].unique() if
                       c in iso_map] if 'country' in buffer_cities.columns else []

        home_country = known_region_cities.iloc[0]['country'] if 'country' in known_region_cities.columns else 'Spain'
        home_iso = iso_map.get(home_country, 'ES')
        if home_iso not in target_isos: target_isos.append(home_iso)
        if 'ES' in target_isos and 'FR' in target_isos and 'AD' not in target_isos: target_isos.append('AD')

        patrol_dfs = []
        for iso in set(target_isos):
            try:
                patrol_dfs.append(self.get_regional_data(country_code=iso, region_bounds=patrol_bounds))
            except:
                pass

        full_patrol_df = pd.concat(patrol_dfs).reset_index(drop=True)

        anchor_lat, anchor_lon = known_region_cities.iloc[0]['latitude'], known_region_cities.iloc[0]['longitude']
        home_df = full_patrol_df[full_patrol_df['country_code'] == home_iso].reset_index(drop=True)
        distances = (home_df['latitude'] - anchor_lat) ** 2 + (home_df['longitude'] - anchor_lon) ** 2
        target_admin_code = home_df.loc[distances.idxmin(), 'admin1_code']

        country_coords = np.array(
            [[np.radians(row['latitude']), np.radians(row['longitude'])] for _, row in full_patrol_df.iterrows()])
        country_admin_codes, country_iso_codes = full_patrol_df['admin1_code'].values, full_patrol_df[
            'country_code'].values
        border_tree = BallTree(country_coords, metric='haversine')
        # ----------------------------------------------

        # 2. Setup the Unified Physics Environment
        total_pop = sum(self.G.nodes[h].get('pop', 10000) for h in hubs)
        num_users = min(2500,
                        max(100, int(total_pop / pop_scale)))  # Cap at 2500 so the physics engine doesn't melt your PC
        print(f"Spawning {num_users} users across the unified backbone...")

        H = nx.Graph()
        initial_pos = {}
        for h in hubs:
            H.add_node(h)
            initial_pos[h] = self.G.nodes[h]['pos']

        # Inject existing backbone edges into the physics engine so it feels the tension between cities
        for u, v in self.G.edges(hubs):
            if u in hubs and v in hubs:
                H.add_edge(u, v)

        active_nodes = list(hubs)
        new_nodes = []

        # 3. MACRO-SCALE DUAL BARABÁSI-ALBERT
        for i in range(num_users):
            user_id = f"T4_MacroUser_{i}"
            H.add_node(user_id)
            new_nodes.append(user_id)

            m = m1 if random.random() < p else m2
            degrees = np.array([H.degree(t) for t in active_nodes])
            probs = degrees / degrees.sum() if degrees.sum() > 0 else np.ones(len(active_nodes)) / len(active_nodes)

            chosen = np.random.choice(active_nodes, size=min(m, len(active_nodes)), replace=False, p=probs)
            for c in chosen:
                H.add_edge(user_id, c, edge_type='T4_Access')

            active_nodes.append(user_id)

            # Give it a starting point near its primary connection so the physics engine doesn't explode
            root_pos = initial_pos[chosen[0]]
            initial_pos[user_id] = (root_pos[0] + random.uniform(-0.05, 0.05),
                                    root_pos[1] + random.uniform(-0.05, 0.05))

        # 4. UNIFIED PHYSICS ENGINE
        print("Running macro-physics simulation. (This may take 5-10 seconds to settle thousands of nodes...)")
        # 'k' dictates ideal spacing. The hubs are completely fixed in place.
        final_pos = nx.spring_layout(H, pos=initial_pos, fixed=hubs, k=0.018, iterations=40)

        # 5. GEOGRAPHY & CUSTOMS APPLICATION
        print("Applying geography and border constraints...")
        for n in new_nodes:
            px, py = final_pos[n]

            # Find its closest anchor for the pull-back string
            closest_hub = list(H.neighbors(n))[0]
            hub_lon, hub_lat = initial_pos[closest_hub]

            # Ocean Check
            angle = 0
            while not globe.is_land(py, px) and angle < 360:
                angle += 15
                theta = np.radians(angle)
                rel_x, rel_y = px - hub_lon, py - hub_lat
                px = hub_lon + (np.cos(theta) * rel_x) - (np.sin(theta) * rel_y)
                py = hub_lat + (np.sin(theta) * rel_x) + (np.cos(theta) * rel_y)

            # Strict Border Patrol
            attempts = 0
            while attempts < 20:
                pt = np.array([[np.radians(py), np.radians(px)]])
                _, idx = border_tree.query(pt, k=1)

                if country_admin_codes[idx[0][0]] == target_admin_code and country_iso_codes[idx[0][0]] == home_iso:
                    break

                    # Pull back towards its anchor
                px = hub_lon + (px - hub_lon) * 0.85
                py = hub_lat + (py - hub_lat) * 0.85
                attempts += 1

            self.G.add_node(n, pos=(px, py), type='Tier4_User', region=target_regions[0])
            for neighbor in H.neighbors(n):
                if not self.G.has_edge(n, neighbor):
                    self.G.add_edge(n, neighbor, edge_type='T4_Access')

        print(f"Unified Tier 4 Complete. Wove {num_users} users into the regional continuum.")
        return self.G

    def plot_network(self):
        print("Generating Folium Map...")
        m = folium.Map(location=[40.4, -3.7], zoom_start=6, tiles="CartoDB dark_matter", prefer_canvas=True)

        for u, v, edge_data in self.G.edges(data=True):
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']
            e_color = edge_data.get('color', '#808080')
            e_type = edge_data.get('edge_type', 'Unknown')

            # Make Tier 1 & 2 highly visible, dim the user access links
            e_weight = 2.5 if 'T1' in e_type else (1.5 if 'T2' in e_type else 0.8)
            e_opacity = 0.9 if 'T1' in e_type else (0.6 if 'T2' in e_type else 0.3)

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=e_color, weight=e_weight, opacity=e_opacity, popup=f"Type: {e_type}"
            ).add_to(m)

        for node_id, node_data in self.G.nodes(data=True):
            lon, lat = node_data['pos']
            n_color = node_data.get('color', '#FFFFFF')
            n_type = node_data.get('type', 'Unknown')
            n_radius = 5 if 'Tier1' in n_type else (3 if 'Tier2' in n_type else (2 if 'Tier3' in n_type else 1))

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
        m.save("quantum_network_app3.html")
        print("Saved as quantum_network_app3.html")

    def plot_network_top(self):
        print("Generating Fully Topological Folium Map...")
        m = folium.Map(location=[40.4, -3.7], zoom_start=6, tiles="CartoDB dark_matter", prefer_canvas=True)

        # 1. Draw edges (Now with Dynamic Topology Colors!)
        for u, v, edge_data in self.G.edges(data=True):
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']
            e_type = edge_data.get('edge_type', 'Unknown')

            # --- DYNAMIC EDGE COLORING FOR TIER 4 ---
            if 'T4' in e_type or 'City_Access' in e_type:
                deg_u = self.G.degree(u)
                deg_v = self.G.degree(v)

                # If it touches a leaf/end-user -> YELLOW
                if deg_u == 1 or deg_v == 1:
                    e_color = '#FFFF00'
                # If it connects two heavy switches -> RED
                elif deg_u >= 3 and deg_v >= 3:
                    e_color = '#FF0000'
                # Otherwise, it's a repeater chain -> BLUE
                else:
                    e_color = '#1E90FF'
            else:
                # Keep original colors for Tiers 1-3
                e_color = edge_data.get('color', '#808080')

            e_weight = 2.5 if 'T1' in e_type else (1.5 if 'T2' in e_type else (1.0 if 'T3' in e_type else 0.5))
            e_opacity = 0.9 if 'T1' in e_type else (0.6 if 'T2' in e_type else (0.4 if 'T3' in e_type else 0.3))

            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color=e_color, weight=e_weight, opacity=e_opacity,
                popup=f"Link: {e_type}"
            ).add_to(m)

        # 2. Draw nodes and classify topology
        for node_id, node_data in self.G.nodes(data=True):
            lon, lat = node_data['pos']
            n_type = node_data.get('type', 'Unknown')

            degree = self.G.degree(node_id)
            if degree == 1:
                topo_role = "End Node (Degree 1)"
            elif degree == 2:
                topo_role = "Repeater (Degree 2)"
            else:
                topo_role = f"Switch (Degree {degree})"

            if 'Tier4' in n_type:
                # --- CORRECTED NODE COLORS ---
                if degree == 1:
                    n_color = '#FFFF00'  # Yellow for End Users
                elif degree == 2:
                    n_color = '#1E90FF'  # Blue for Repeaters
                else:
                    n_color = '#FF0000'  # Red for Switches
                n_radius = 1.5
            else:
                n_color = node_data.get('color', '#FFFFFF')
                n_radius = 5 if 'Tier1' in n_type else (3 if 'Tier2' in n_type else 2)

            popup_html = f"<b>{node_id}</b><br>Tier: {n_type}<br><b>Topology: {topo_role}</b>"

            folium.CircleMarker(
                location=[lat, lon], radius=n_radius, color=n_color, fill=True,
                fill_color=n_color, fill_opacity=0.9, popup=popup_html
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
        m.save("quantum_network_topological.html")
        print("Saved as quantum_network_topological.html")

    def get_regional_data(self, country_code='ES', region_bounds=None):
        """
        Fetches fresh data if needed and filters it for the specific region.
        region_bounds: tuple (min_lat, max_lat, min_lon, max_lon)
        """
        txt_file = f"{country_code}.txt"
        zip_file = f"{country_code}.zip"

        # 1. Download/Load logic
        if not os.path.exists(txt_file):
            print(f"Fetching GeoNames data for {country_code}...")
            url = f"https://download.geonames.org/export/dump/{zip_file}"
            response = requests.get(url)
            response.raise_for_status()

            # Extract the .txt from the .zip in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(txt_file) as f:
                    with open(txt_file, 'wb') as out:
                        out.write(f.read())

        # 2. Load into DataFrame
        df = pd.read_csv(txt_file, sep='\t', header=None,
                         names=['geonameid', 'name', 'asciiname', 'alternatenames',
                                'latitude', 'longitude', 'feature_class', 'feature_code',
                                'country_code', 'cc2', 'admin1_code', 'admin2_code',
                                'admin3_code', 'admin4_code', 'population', 'elevation',
                                'dem', 'timezone', 'modification_date'],
                         dtype={'admin1_code': str, 'admin2_code': str},
                         low_memory=False)

        # 3. Apply region filter
        if region_bounds:
            min_lat, max_lat, min_lon, max_lon = region_bounds
            df = df[
                (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
                ]

        # 4. Filter for populated places only
        df = df[df['feature_class'] == 'P']

        return df

    def generate_tier3_regional(self, target_regions=['Catalonia'], gap_km=35, min_pop=1500):
        print(f"Tier 3: Generating Organic Regional Backhaul in {target_regions}...")

        hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('region') in target_regions and (
                'Tier1' in attr.get('type', '') or 'Tier2' in attr.get('type', ''))]
        active_t3_nodes = []
        active_t3_coords = []

        all_active_nodes = list(self.G.nodes())
        all_active_coords = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in all_active_nodes])
        hub_coords_rad = np.array(
            [[np.radians(self.G.nodes[n]['pos'][1]), np.radians(self.G.nodes[n]['pos'][0])] for n in hubs])

        # --- COMPLETELY DYNAMIC COUNTRY DETECTION ---
        # 1. Find the known cities in your CSV for the target region
        known_region_cities = self.cities[self.cities['region'] == target_regions[0]]
        if known_region_cities.empty:
            print(f"Error: No known cities in {target_regions[0]} to anchor the search.")
            return self.G

        # 2. Map the detected country to its ISO code
        iso_map = {
            'Spain': 'ES', 'France': 'FR', 'Andorra': 'AD', 'Portugal': 'PT',
            'Germany': 'DE', 'Italy': 'IT', 'United Kingdom': 'GB', 'Belgium': 'BE',
            'Netherlands': 'NL', 'Switzerland': 'CH', 'Austria': 'AT', 'Poland': 'PL',
            'Sweden': 'SE', 'Norway': 'NO', 'Denmark': 'DK', 'Finland': 'FI', 'Ireland': 'IE'
        }

        home_country = known_region_cities.iloc[0]['country'] if 'country' in known_region_cities.columns else 'Spain'
        home_iso = iso_map.get(home_country, 'ES')  # Default to ES if not found

        print(f"Detected home country: {home_country} ({home_iso})")

        # 3. Get the entire country dataset dynamically
        full_country_df = self.get_regional_data(country_code=home_iso)

        # 4. Find the admin1_code using a SPATIAL ANCHOR (Immune to name mismatches!)
        anchor_lat = known_region_cities.iloc[0]['latitude']
        anchor_lon = known_region_cities.iloc[0]['longitude']

        # Find the geometrically closest point in the GeoNames database
        distances = (full_country_df['latitude'] - anchor_lat) ** 2 + (full_country_df['longitude'] - anchor_lon) ** 2
        closest_geoname_idx = distances.idxmin()

        # 5. Extract the official region code from that physical location
        region_code = full_country_df.loc[closest_geoname_idx, 'admin1_code']
        geoname_anchor_name = full_country_df.loc[closest_geoname_idx, 'name']

        print(f"Anchored '{target_regions[0]}' via {geoname_anchor_name} -> admin1_code: {region_code}")

        # 6. Filter the millions of points down to ONLY this exact region
        geo_df = full_country_df[full_country_df['admin1_code'] == region_code]
        geo_df = geo_df[geo_df['population'] >= min_pop].sort_values(by='population', ascending=False)

        added_count = 0
        for _, row in geo_df.iterrows():
            lat, lon, pop, name = row['latitude'], row['longitude'], row['population'], row['name']

            if name in self.G.nodes():
                continue

            pt = np.array([[np.radians(lat), np.radians(lon)]])
            dists_to_all = haversine_distances(pt, all_active_coords)[0] * 6371

            if np.min(dists_to_all) > gap_km:
                self.G.add_node(name, pos=(lon, lat), type='Tier3_Real', region=target_regions[0], pop=pop,
                                color='#00FF00')
                added_count += 1

                # WIRING A: Uplink to closest T1/T2 Hub
                if len(hubs) > 0:
                    dists_to_hubs = haversine_distances(pt, hub_coords_rad)[0] * 6371
                    closest_hub = hubs[np.argmin(dists_to_hubs)]
                    self.G.add_edge(name, closest_hub, weight=np.min(dists_to_hubs), edge_type='T3_Uplink',
                                    color='#00FF00')

                # WIRING B: True Starburst Preferential Attachment (The Tier 2 Math)
                if len(active_t3_nodes) > 0:
                    t3_coords_arr = np.vstack(active_t3_coords)
                    dists_to_t3 = haversine_distances(pt, t3_coords_arr)[0] * 6371
                    valid_indices = np.where(dists_to_t3 < 120.0)[0]  # Wider search radius to find hubs

                    if len(valid_indices) > 0:
                        weights = [(self.G.degree(active_t3_nodes[v]) + 1) / (dists_to_t3[v] + 1) for v in
                                   valid_indices]
                        probs = [w / sum(weights) for w in weights]

                        # size=1 ensures it acts like a tree branch, not a fishing net
                        chosen = np.random.choice(valid_indices, size=1, p=probs)

                        for c_idx in chosen:
                            self.G.add_edge(name, active_t3_nodes[c_idx], weight=dists_to_t3[c_idx],
                                            edge_type='T3_Mesh', color='#00FF00')

                active_t3_nodes.append(name)
                active_t3_coords = pt if len(active_t3_coords) == 0 else np.vstack([active_t3_coords, pt])
                all_active_nodes.append(name)
                all_active_coords = np.vstack([all_active_coords, pt])

        print(f"Tier 3 Complete. Added {added_count} real organic distribution nodes.")
        return self.G
# 1. Load your newly generated CSV
df = pd.read_csv('ultimate_city_coordinates.csv')

# Initialize the builder
net3 = QuantumNetworkBuilder_App3(df)

# 1. Continental Core
net3.generate_tier1()

# 2. National Skeleton (Spain and Germany)
net3.generate_tier2(target_countries=['Spain'], gap_km=80)
#
# 3 & 4. Regional and Fractal Networks (Run them ONE BY ONE)
regions_to_map = ['Catalonia']

for region in regions_to_map:
    print(f"\n--- INITIATING BUILD FOR: {region.upper()} ---")
    net3.generate_tier3_regional(target_regions=[region])
    net3.generate_tier4_fractal(target_regions=[region], pop_scale=500)

# 5. Plot with the Dynamic Edge Colors
net3.plot_network_top()

