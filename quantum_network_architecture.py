
import pandas as pd
import numpy as np
import networkx as nx
import random
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances
import folium
from branca.element import Element
import os
import requests
import zipfile
import io
from global_land_mask import globe


class QuantumNetworkBuilder:
    def __init__(self, cities_df, type = 'hybrid'):
        self.cities = cities_df
        self.G = nx.Graph()
        self.type = type

        # Categorize the real cities
        self.tier1_df = cities_df[cities_df['population'] >= 1000000].copy()
        self.tier2_df = cities_df[(cities_df['population'] < 1000000) & (cities_df['population'] > 100000)].copy()
        self.tier3_df = cities_df[cities_df['population'] <= 100000].copy()

        if type == 'hybrid':
            self.d_c_CV = 92.84
            self.d_cross = 91.13
            self.d_c_DV = 354.82

        elif type == 'entanglement':
            self.d_max = 250 # This will probably depend a bit on number of hops or something. Depends on other code

    def generate_tier1(self, mode='satellite'):
        # 1. Determine the base threshold distance (Fibre range)
        if self.type == 'hybrid':
            base_d_max = self.d_c_DV  # This will successfully pull your 354.82
        else:
            base_d_max = self.d_max  # This will successfully pull your 250

        # 2. Determine relay placement distance
        if mode == 'satellite':
            relay_d_max = base_d_max * 2
        else:
            relay_d_max = base_d_max

        print(f"Tier 1: Generating Continental Core Backbone ({mode.upper()} Mode)...")

        # --- PHASE 1: NODE PLACEMENT ---
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

            # Use the dynamic relay distance here
            if np.min(dists) > relay_d_max:
                self.G.add_node(row['city_name'], pos=(row['longitude'], row['latitude']),
                                type='Tier1_Relay', pop=row['population'], region=row.get('region', 'Unknown'),
                                color='#9400D3')
                active_hubs_rad = np.vstack([active_hubs_rad, city_rad])

        # --- PHASE 2: TOPOLOGY WIRING ---
        # Calculate the distance matrix BEFORE drawing cables
        node_names = list(self.G.nodes())
        final_coords_deg = np.array([[self.G.nodes[n]['pos'][1], self.G.nodes[n]['pos'][0]] for n in node_names])
        final_coords_rad = np.radians(final_coords_deg)
        dist_matrix = haversine_distances(final_coords_rad, final_coords_rad) * 6371

        if mode == 'old':
            # Classic MST + KNN mesh
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

        elif mode in ['fibre', 'middleground', 'enhanced_middleground', 'satellite', 'backbone_spoke',
                      'fibre_covered_backbone', 'repeater_middleground']:
            # A. THE FIBRE LAYER (Common to all these modes)
            for i in range(len(node_names)):
                for j in range(i + 1, len(node_names)):
                    if dist_matrix[i, j] <= base_d_max:
                        self.G.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j],
                                        edge_type='Fibre_Link', color='#FFFFFF')

            # B. THE SATELLITE LAYER LOGIC
            if mode == 'fibre':
                pass  # Baseline, no satellites

            elif mode == 'middleground':
                # MST Island Bridge
                G_temp = nx.Graph()
                for i in range(len(node_names)):
                    for j in range(i + 1, len(node_names)):
                        G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])
                mst = nx.minimum_spanning_tree(G_temp, weight='weight')
                for u, v, data in mst.edges(data=True):
                    if not self.G.has_edge(u, v):
                        self.G.add_edge(u, v, weight=data['weight'], edge_type='Satellite_Link', color='#FFD700')

                # Geographically Distributed Regional Anchors
                hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Hub']
                hub_degrees = {n: self.G.degree(n) for n in hubs}
                sorted_hubs = sorted(hub_degrees, key=hub_degrees.get, reverse=True)
                regional_anchors = []
                for hub in sorted_hubs:
                    hub_idx = node_names.index(hub)
                    is_far_enough = all(dist_matrix[hub_idx, node_names.index(a)] >= 700 for a in regional_anchors)
                    if is_far_enough: regional_anchors.append(hub)
                for i in range(len(regional_anchors)):
                    for j in range(i + 1, len(regional_anchors)):
                        u, v = regional_anchors[i], regional_anchors[j]
                        if not self.G.has_edge(u, v):
                            self.G.add_edge(u, v, weight=dist_matrix[node_names.index(u), node_names.index(v)],
                                            edge_type='Satellite_Link', color='#FFD700')

            elif mode == 'enhanced_middleground':
                # 1. Start with the standard middleground logic (MST + Anchors)
                G_temp = nx.Graph()
                for i in range(len(node_names)):
                    for j in range(i + 1, len(node_names)):
                        G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])
                mst = nx.minimum_spanning_tree(G_temp, weight='weight')
                for u, v, data in mst.edges(data=True):
                    if not self.G.has_edge(u, v):
                        self.G.add_edge(u, v, weight=data['weight'], edge_type='Satellite_Link', color='#FFD700')

                hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Hub']
                hub_degrees = {n: self.G.degree(n) for n in hubs}
                sorted_hubs = sorted(hub_degrees, key=hub_degrees.get, reverse=True)
                biggest_hub = sorted_hubs[0] if sorted_hubs else node_names[0]

                regional_anchors = []
                for hub in sorted_hubs:
                    hub_idx = node_names.index(hub)
                    if all(dist_matrix[hub_idx, node_names.index(a)] >= 700 for a in regional_anchors):
                        regional_anchors.append(hub)

                # 2. Connect Regional Anchors to each other AND to the "King Hub"
                for anchor in regional_anchors:
                    if anchor != biggest_hub and not self.G.has_edge(anchor, biggest_hub):
                        self.G.add_edge(anchor, biggest_hub,
                                        weight=dist_matrix[node_names.index(anchor), node_names.index(biggest_hub)],
                                        edge_type='Satellite_Link', color='#FFD700')

                # 3. Nearest Neighbor guarantee: Every node connects to its absolute closest neighbor
                for i in range(len(node_names)):
                    nearest_idx = np.argsort(dist_matrix[i])[1]  # Index 0 is itself, Index 1 is closest
                    u, v = node_names[i], node_names[nearest_idx]
                    if not self.G.has_edge(u, v):
                        self.G.add_edge(u, v, weight=dist_matrix[i, nearest_idx], edge_type='Satellite_Link',
                                        color='#FFD700')

            elif mode == 'repeater_middleground':
                # 1. Same logic as middleground: MST to find the critical bridges
                G_temp = nx.Graph()
                for i in range(len(node_names)):
                    for j in range(i + 1, len(node_names)):
                        G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])
                mst = nx.minimum_spanning_tree(G_temp, weight='weight')

                # We will define a helper function to inject repeaters along a line
                def inject_repeaters(u, v, total_dist):
                    if total_dist <= base_d_max:
                        self.G.add_edge(u, v, weight=total_dist, edge_type='Fibre_Link', color='#FFFFFF')
                        return

                    # Calculate how many repeaters are needed to keep gaps <= base_d_max
                    num_repeaters = int(np.ceil(total_dist / base_d_max)) - 1
                    u_lon, u_lat = self.G.nodes[u]['pos']
                    v_lon, v_lat = self.G.nodes[v]['pos']

                    prev_node = u
                    step_dist = total_dist / (num_repeaters + 1)

                    for r in range(1, num_repeaters + 1):
                        frac = r / (num_repeaters + 1)
                        # Linear interpolation for coordinates
                        r_lon = u_lon + frac * (v_lon - u_lon)
                        r_lat = u_lat + frac * (v_lat - u_lat)
                        r_name = f"Repeater_{u[:3]}_{v[:3]}_{r}"

                        self.G.add_node(r_name, pos=(r_lon, r_lat), type='Tier1_Repeater', color='#00FF00')
                        self.G.add_edge(prev_node, r_name, weight=step_dist, edge_type='Fibre_Link', color='#FFFFFF')
                        prev_node = r_name

                    self.G.add_edge(prev_node, v, weight=step_dist, edge_type='Fibre_Link', color='#FFFFFF')

                # Inject repeaters instead of satellites for the MST
                for u, v, data in mst.edges(data=True):
                    if not self.G.has_edge(u, v):
                        inject_repeaters(u, v, data['weight'])

                # 2. Geographically Distributed Anchors
                hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Hub']
                hub_degrees = {n: self.G.degree(n) for n in hubs}
                sorted_hubs = sorted(hub_degrees, key=hub_degrees.get, reverse=True)
                regional_anchors = []
                for hub in sorted_hubs:
                    hub_idx = node_names.index(hub)
                    if all(dist_matrix[hub_idx, node_names.index(a)] >= 700 for a in regional_anchors):
                        regional_anchors.append(hub)

                # Inject repeaters between regional anchors
                for i in range(len(regional_anchors)):
                    for j in range(i + 1, len(regional_anchors)):
                        u, v = regional_anchors[i], regional_anchors[j]
                        if not self.G.has_edge(u, v):
                            inject_repeaters(u, v, dist_matrix[node_names.index(u), node_names.index(v)])

            elif mode == 'satellite':
                # 1. Bridge the isolated components
                G_temp = nx.Graph()
                for i in range(len(node_names)):
                    for j in range(i + 1, len(node_names)):
                        G_temp.add_edge(node_names[i], node_names[j], weight=dist_matrix[i, j])
                mst = nx.minimum_spanning_tree(G_temp, weight='weight')
                for u, v, data in mst.edges(data=True):
                    if not self.G.has_edge(u, v):
                        self.G.add_edge(u, v, weight=data['weight'], edge_type='Satellite_Link', color='#FF8C00')

                # 2. Aggressive Small World (The Mega-Hub Constellation)
                mega_hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Hub']
                relays = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Relay']

                for i in range(len(mega_hubs)):
                    for j in range(i + 1, len(mega_hubs)):
                        if not self.G.has_edge(mega_hubs[i], mega_hubs[j]):
                            dist = dist_matrix[node_names.index(mega_hubs[i]), node_names.index(mega_hubs[j])]
                            self.G.add_edge(mega_hubs[i], mega_hubs[j], weight=dist, edge_type='Satellite_Link',
                                            color='#FF8C00')

                # 3. Ensure every Relay has direct access to the Constellation
                for relay in relays:
                    relay_idx = node_names.index(relay)
                    mega_hub_indices = [node_names.index(mh) for mh in mega_hubs]

                    if mega_hub_indices:
                        dists_to_mhs = dist_matrix[relay_idx, mega_hub_indices]
                        closest_mh_idx = mega_hub_indices[np.argmin(dists_to_mhs)]
                        closest_mh = node_names[closest_mh_idx]
                        if not self.G.has_edge(relay, closest_mh):
                            dist = dist_matrix[relay_idx, closest_mh_idx]
                            self.G.add_edge(relay, closest_mh, weight=dist, edge_type='Satellite_Link', color='#FF8C00')

            elif mode == 'backbone_spoke':
                # 1. Build the Elite Backbone (Regional anchors at 500km apart for a bit more density)
                hubs = [n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'Tier1_Hub']
                hub_pops = {n: self.G.nodes[n].get('pop', 0) for n in hubs}
                sorted_hubs = sorted(hub_pops, key=hub_pops.get, reverse=True)

                backbone_nodes = []
                for hub in sorted_hubs:
                    hub_idx = node_names.index(hub)
                    if all(dist_matrix[hub_idx, node_names.index(b)] >= 500 for b in backbone_nodes):
                        backbone_nodes.append(hub)

                # Fully mesh the backbone via satellite
                for i in range(len(backbone_nodes)):
                    for j in range(i + 1, len(backbone_nodes)):
                        u, v = backbone_nodes[i], backbone_nodes[j]
                        if not self.G.has_edge(u, v):
                            self.G.add_edge(u, v, weight=dist_matrix[node_names.index(u), node_names.index(v)],
                                            edge_type='Satellite_Link', color='#FF8C00')

                # 2. Every other node acts as a spoke and connects to the closest backbone node
                backbone_indices = [node_names.index(b) for b in backbone_nodes]
                for i, node in enumerate(node_names):
                    if node not in backbone_nodes:
                        dists_to_backbone = dist_matrix[i, backbone_indices]
                        closest_bb_idx = backbone_indices[np.argmin(dists_to_backbone)]
                        closest_bb_node = node_names[closest_bb_idx]

                        if not self.G.has_edge(node, closest_bb_node):
                            dist = dist_matrix[i, closest_bb_idx]
                            # If it's close enough, it's fibre. If not, it's satellite.
                            edge_color = '#FFFFFF' if dist <= base_d_max else '#FFD700'
                            edge_type = 'Fibre_Link' if dist <= base_d_max else 'Satellite_Link'
                            self.G.add_edge(node, closest_bb_node, weight=dist, edge_type=edge_type, color=edge_color)

            elif mode == 'fibre_covered_backbone':
                # 1. Build a "Dominating Set" Backbone
                # We iteratively pick the highest population nodes. If a node is picked,
                # all nodes within base_d_max of it are "covered" and don't need to be in the backbone.
                sorted_all_nodes = sorted(self.G.nodes(data=True), key=lambda x: x[1].get('pop', 0), reverse=True)
                backbone_nodes = []
                covered_indices = set()

                for n, attr in sorted_all_nodes:
                    n_idx = node_names.index(n)
                    if n_idx not in covered_indices:
                        backbone_nodes.append(n)
                        covered_indices.add(n_idx)
                        # Mark all nodes within fibre range as covered
                        for j in range(len(node_names)):
                            if dist_matrix[n_idx, j] <= base_d_max:
                                covered_indices.add(j)

                # 2. Mesh the Backbone together (MST + K=2 Nearest Backbone Neighbors for redundancy)
                backbone_indices = [node_names.index(b) for b in backbone_nodes]
                G_bb = nx.Graph()
                for i in range(len(backbone_indices)):
                    for j in range(i + 1, len(backbone_indices)):
                        G_bb.add_edge(backbone_nodes[i], backbone_nodes[j],
                                      weight=dist_matrix[backbone_indices[i], backbone_indices[j]])

                bb_mst = nx.minimum_spanning_tree(G_bb, weight='weight')
                for u, v, data in bb_mst.edges(data=True):
                    if not self.G.has_edge(u, v):
                        self.G.add_edge(u, v, weight=data['weight'], edge_type='Satellite_Link', color='#FF8C00')

                # 3. Connect the spokes: EVERY non-backbone node is guaranteed to be <= base_d_max to a backbone node
                for i, node in enumerate(node_names):
                    if node not in backbone_nodes:
                        dists_to_bb = dist_matrix[i, backbone_indices]
                        closest_bb_idx = backbone_indices[np.argmin(dists_to_bb)]
                        closest_bb_node = node_names[closest_bb_idx]

                        if not self.G.has_edge(node, closest_bb_node):
                            dist = dist_matrix[i, closest_bb_idx]
                            # Guaranteed to be Fibre because of how we selected the backbone!
                            self.G.add_edge(node, closest_bb_node, weight=dist, edge_type='Fibre_Link', color='#FFFFFF')

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

    def generate_tier2(self, target_countries=['Spain', 'United Kingdom'], gap_km=None, min_pop_geonames=50000):
        # --- PHYSICS PARAMETER SETUP ---
        if self.type == 'hybrid':
            actual_gap_km = gap_km if gap_km else self.d_c_CV
            dv_limit = self.d_c_DV
            cv_limit = self.d_c_CV
        else:
            actual_gap_km = gap_km if gap_km else 80.0
            dv_limit = self.d_max
            cv_limit = 0  # No CV in pure entanglement mode

        # Helper function to classify edges by physics limits
        def get_quantum_link_props(dist, layer_prefix):
            if cv_limit > 0 and dist <= cv_limit:
                return f'{layer_prefix}_CV', '#00FF00'  # Green: High-Bandwidth CV-QKD
            elif dist <= dv_limit:
                return f'{layer_prefix}_DV', '#0000FF'  # Blue: Long-Range DV-QKD
            else:
                return f'{layer_prefix}_Satellite', '#FFD700'  # Yellow: Space-based fallback

        print(f"Tier 2: Generating Density-Controlled National Backhaul ({target_countries})...")

        # --- DYNAMIC COUNTRY CODE MAPPING ---
        # 1. Comprehensive fallback dictionary for Europe
        country_code_map = {
            'Albania': 'AL', 'Andorra': 'AD', 'Austria': 'AT', 'Belarus': 'BY',
            'Belgium': 'BE', 'Bosnia and Herzegovina': 'BA', 'Bulgaria': 'BG',
            'Croatia': 'HR', 'Cyprus': 'CY', 'Czechia': 'CZ', 'Czech Republic': 'CZ',
            'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR',
            'Germany': 'DE', 'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS',
            'Ireland': 'IE', 'Italy': 'IT', 'Kosovo': 'XK', 'Latvia': 'LV',
            'Liechtenstein': 'LI', 'Lithuania': 'LT', 'Luxembourg': 'LU',
            'Malta': 'MT', 'Moldova': 'MD', 'Monaco': 'MC', 'Montenegro': 'ME',
            'Netherlands': 'NL', 'North Macedonia': 'MK', 'Norway': 'NO',
            'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Russia': 'RU',
            'San Marino': 'SM', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI',
            'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR',
            'Ukraine': 'UA', 'United Kingdom': 'GB', 'Vatican City': 'VA'
        }

        # 2. Extract mappings dynamically from dataset to guarantee we have what we want
        try:
            # Safely attempt to gather country and country_code from your actual data
            all_cities = pd.concat([
                getattr(self, 'tier1_df', pd.DataFrame()),
                getattr(self, 'tier2_df', pd.DataFrame()),
                getattr(self, 'tier3_df', pd.DataFrame())
            ])
            if 'country' in all_cities.columns and 'country_code' in all_cities.columns:
                dynamic_map = dict(zip(all_cities['country'], all_cities['country_code']))
                country_code_map.update(dynamic_map)
        except Exception as e:
            print(f"Note: Could not dynamically build country codes, using fallback list. ({e})")

        # --- STEP 1: Base candidates from Eurostat ---
        candidates = pd.concat([
            self.tier2_df[self.tier2_df['country'].isin(target_countries)],
            self.tier3_df[self.tier3_df['country'].isin(target_countries)]
        ]).sort_values(by='population', ascending=False)
        candidates = candidates[~candidates['city_name'].isin(self.G.nodes())]

        # --- STEP 2: Load GeoNames as a supplement ---
        geonames_supplements = []
        for country in target_countries:
            cc = country_code_map.get(country)
            if not cc:
                print(f"Warning: Could not find country code for '{country}'. Skipping GeoNames expansion.")
                continue
            gdf = self.get_regional_data(country_code=cc)
            gdf['country'] = country
            gdf = gdf.rename(columns={'name': 'city_name'})
            geonames_supplements.append(gdf)

        if geonames_supplements:
            geonames_df = pd.concat(geonames_supplements).sort_values(by='population', ascending=False)
            geonames_df = geonames_df[geonames_df['population'] >= min_pop_geonames]

            eurostat_coords_rad = np.array([
                [np.radians(row['latitude']), np.radians(row['longitude'])]
                for _, row in candidates.iterrows()
            ])

            if len(eurostat_coords_rad) > 0:
                eurostat_tree = BallTree(eurostat_coords_rad, metric='haversine')

                gap_fill_rows = []
                for _, row in geonames_df.iterrows():
                    if row['city_name'] in candidates['city_name'].values:
                        continue
                    pt = np.array([[np.radians(row['latitude']), np.radians(row['longitude'])]])
                    dist_rad, _ = eurostat_tree.query(pt, k=1)
                    dist_km = dist_rad[0][0] * 6371
                    if dist_km > actual_gap_km:
                        gap_fill_rows.append(row)

                if gap_fill_rows:
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
                        indices = gap_tree.query_radius(pt, r=actual_gap_km / 6371)
                        for idx in indices[0]:
                            if idx != i:
                                used_indices.add(idx)
                        deduped.append(row)

                    gap_fill_df = pd.DataFrame(deduped)
                    candidates = pd.concat([candidates, gap_fill_df]).sort_values(by='population', ascending=False)

        # --- STEP 3: Connect the nodes ---
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

            if pop > 100000 or min_dist > actual_gap_km:
                self.G.add_node(name, pos=(row['longitude'], row['latitude']), type='Tier2_Aggregator',
                                pop=pop, region=row.get('region', row.get('admin1_code', 'Unknown')), color='#FFA500')

                # Connect to nearest Tier 1
                if len(active_t1_nodes) > 0:
                    dists_to_t1 = haversine_distances(new_rad, active_t1_coords)[0] * 6371
                    nearest_t1_idx = np.argmin(dists_to_t1)
                    t1_target = active_t1_nodes[nearest_t1_idx]
                    t1_dist = dists_to_t1[nearest_t1_idx]

                    e_type, e_color = get_quantum_link_props(t1_dist, 'T2_to_T1')
                    self.G.add_edge(name, t1_target, weight=t1_dist, edge_type=e_type, color=e_color)

                # Connect to other Tier 2s (Physics limit: strictly <= 200km for reliable DV-QKD SNR without repeaters)
                if len(active_t2_nodes) > 0:
                    t2_coords_arr = np.vstack(active_t2_coords)
                    dists_to_t2 = haversine_distances(new_rad, t2_coords_arr)[0] * 6371

                    # 200km strict hardware threshold
                    valid_indices = np.where(dists_to_t2 <= 200.0)[0]

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
                            t2_dist = dists_to_t2[c_idx]
                            e_type, e_color = get_quantum_link_props(t2_dist, 'T2_to_T2')
                            self.G.add_edge(name, active_t2_nodes[c_idx], weight=t2_dist, edge_type=e_type,
                                            color=e_color)

                active_t2_nodes.append(name)
                active_t2_coords = new_rad if len(active_t2_coords) == 0 else np.vstack([active_t2_coords, new_rad])
                all_active_nodes.append(name)
                all_active_coords = np.vstack([all_active_coords, new_rad])

        print(f"Tier 2 Complete. Added {len(active_t2_nodes)} Regional Nodes.")
        return self.G

    def generate_tier3(self, target_regions=['Catalonia'], gap_km=None, min_pop=1500):
        # --- PHYSICS PARAMETER SETUP ---
        if getattr(self, 'type', 'hybrid') == 'hybrid':
            # 35km is the standard optical node spacing for Metropolitan Area Networks (MANs)
            actual_gap_km = gap_km if gap_km else 35.0
            cv_limit = getattr(self, 'd_c_CV', 92.84)
            dv_limit = getattr(self, 'd_c_DV', 354.82)
            # Tier 3 MUST prioritize high-bandwidth, cheap CV-QKD where possible
            regional_max_link = cv_limit
        else:
            actual_gap_km = gap_km if gap_km else 35.0
            cv_limit = 0
            dv_limit = getattr(self, 'd_max', 250)
            regional_max_link = dv_limit

        # Helper function to classify edges by physics limits
        def get_quantum_link_props(dist, layer_prefix):
            if cv_limit > 0 and dist <= cv_limit:
                return f'{layer_prefix}_CV', '#00FF00'  # Green: High-Bandwidth CV-QKD
            elif dist <= dv_limit:
                return f'{layer_prefix}_DV', '#0000FF'  # Blue: Long-Range DV-QKD
            else:
                return f'{layer_prefix}_Satellite', '#FFD700'  # Yellow: Space-based fallback

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

            if np.min(dists_to_all) > actual_gap_km:
                self.G.add_node(name, pos=(lon, lat), type='Tier3_Real', region=target_regions[0], pop=pop,
                                country=home_country, color='#00FF00')
                added_count += 1

                # WIRING A: Uplink to closest T1/T2 Hub
                if len(hubs) > 0:
                    dists_to_hubs = haversine_distances(pt, hub_coords_rad)[0] * 6371
                    min_hub_dist = np.min(dists_to_hubs)
                    closest_hub = hubs[np.argmin(dists_to_hubs)]

                    e_type, e_color = get_quantum_link_props(min_hub_dist, 'T3_Uplink')
                    self.G.add_edge(name, closest_hub, weight=min_hub_dist, edge_type=e_type, color=e_color)

                # WIRING B: True Starburst Preferential Attachment (The Tier 2 Math)
                if len(active_t3_nodes) > 0:
                    t3_coords_arr = np.vstack(active_t3_coords)
                    dists_to_t3 = haversine_distances(pt, t3_coords_arr)[0] * 6371

                    # Enforce the physics-based regional max link limit (CV limit)
                    valid_indices = np.where(dists_to_t3 <= regional_max_link)[0]

                    if len(valid_indices) > 0:
                        weights = [(self.G.degree(active_t3_nodes[v]) + 1) / (dists_to_t3[v] + 1) for v in
                                   valid_indices]
                        probs = [w / sum(weights) for w in weights]

                        # size=1 ensures it acts like a tree branch, not a fishing net
                        chosen = np.random.choice(valid_indices, size=1, p=probs)

                        for c_idx in chosen:
                            t3_dist = dists_to_t3[c_idx]
                            e_type, e_color = get_quantum_link_props(t3_dist, 'T3_Mesh')
                            self.G.add_edge(name, active_t3_nodes[c_idx], weight=t3_dist, edge_type=e_type,
                                            color=e_color)

                active_t3_nodes.append(name)
                active_t3_coords = pt if len(active_t3_coords) == 0 else np.vstack([active_t3_coords, pt])
                all_active_nodes.append(name)
                all_active_coords = np.vstack([all_active_coords, pt])

        print(f"Tier 3 Complete. Added {added_count} real organic distribution nodes.")
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

    def plot_network(self, mode_name="default"):
        import folium
        from folium import Element

        print(f"Generating Folium Map for {mode_name.upper()}...")
        m = folium.Map(location=[40.4, -3.7], zoom_start=6, tiles="CartoDB dark_matter", prefer_canvas=True)

        for u, v, edge_data in self.G.edges(data=True):
            u_lon, u_lat = self.G.nodes[u]['pos']
            v_lon, v_lat = self.G.nodes[v]['pos']
            e_color = edge_data.get('color', '#808080')
            e_type = edge_data.get('edge_type', 'Unknown')

            # Make Tier 1 & 2 highly visible, dim the user access links
            e_weight = 2.5 if 'T1' in e_type else (2.0 if 'T2' in e_type else (1.5 if 'T3' in e_type else 1.0))
            e_opacity = 0.9 if 'T1' in e_type else (0.6 if 'T2' in e_type else 0.3)

            if 'DIAMETER_PATH' in e_type:
                e_weight = e_weight * 2

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

        # Notice how mode_name is injected directly into the defaultName variable here!
        screenshot_js = f"""
                <div style="position: absolute; top: 10px; left: 50px; z-index: 9999;">
                    <button id="capture-btn" style="padding: 10px; background-color: white; border: 2px solid rgba(0,0,0,0.2); border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif;">
                        📸 Take Screenshot
                    </button>
                </div>
                <script>
                    window.addEventListener('load', function() {{
                        var btn = document.getElementById('capture-btn');
                        if (btn) {{
                            btn.addEventListener('click', function() {{
                                var defaultName = "quantum_map_{mode_name}";
                                var filename = prompt("Enter a name for your screenshot (without .png):", defaultName);
                                if (filename) {{
                                    btn.style.display = 'none';
                                    var mapContainer = document.querySelector('.leaflet-container');
                                    html2canvas(mapContainer, {{useCORS: true, allowTaint: false}}).then(function(canvas) {{
                                        var link = document.createElement('a');
                                        link.download = filename + '.png';
                                        link.href = canvas.toDataURL('image/png');
                                        link.click();
                                        btn.style.display = 'block';
                                    }});
                                }}
                            }});
                        }}
                    }});
                </script>
                """
        m.get_root().html.add_child(Element(screenshot_js))

        # Save the file with the specific mode name
        file_name = f"quantum_map_{mode_name}.html"
        m.save(file_name)
        print(f"Saved interactive map as {file_name}")

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


    def evaluate_tier1_modes(self):
        import networkx as nx

        modes_to_test = [
            'fibre', 'middleground', 'repeater_middleground',
            'enhanced_middleground', 'satellite',
            'backbone_spoke', 'fibre_covered_backbone'
        ]

        print("\n--- Running Network Evaluation ---")
        latex_rows = []

        for mode in modes_to_test:
            # 1. Clear the graph to ensure a fresh start for each mode
            self.G.clear()

            # 2. Run the mode
            self.generate_tier1(mode=mode)

            # 3. Calculate Metrics
            N = self.G.number_of_nodes()
            E = self.G.number_of_edges()

            sat_edges = sum(1 for u, v, d in self.G.edges(data=True) if d.get('edge_type') == 'Satellite_Link')
            fib_edges = sum(1 for u, v, d in self.G.edges(data=True) if d.get('edge_type') == 'Fibre_Link')

            avg_k = (2 * E) / N if N > 0 else 0

            # --- CONNECTIVITY & PATH METRICS ---
            components = list(nx.connected_components(self.G))
            largest_cc_size = len(max(components, key=len)) if components else 0
            perc_lcc = (largest_cc_size / N) * 100 if N > 0 else 0

            # Path metrics must be calculated on the largest connected component to avoid crashing
            if components:
                largest_cc = max(components, key=len)
                subgraph = self.G.subgraph(largest_cc)
                if len(subgraph) > 1:
                    L = nx.average_shortest_path_length(subgraph)
                    D = nx.diameter(subgraph)  # The longest shortest path!
                else:
                    L, D = 0, 0
            else:
                L, D = 0, 0

            # 4. Format row for LaTeX
            clean_name = mode.replace('_', '\\_').title()
            row = f"        {clean_name} & {N} & {fib_edges} & {sat_edges} & {avg_k:.2f} & {L:.2f} & {D} & {perc_lcc:.1f}\\% \\\\"
            latex_rows.append(row)

            # 5. Automatically generate the Folium map screenshot file for this mode
            if hasattr(self, 'plot_network'):
                self.plot_network(mode_name=mode)

        # 6. Compile and Print the LaTeX Table
        latex_table = f"""
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Quantitative comparison of topological metrics across the generated Tier 1 network modes. ($N$ = Total Nodes, $L$ = Average Shortest Path Length, $D$ = Maximum Shortest Path / Diameter, \\% LCC = Percentage of nodes in the main backbone). Path metrics are calculated for the largest connected component.}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lrrrrrrr}}
        \\toprule
        Graph Mode & $N$ & Fibre Edges & Sat Edges & $\\langle k \\rangle$ & Avg $L$ & Max $L$ ($D$) & \\% LCC \\\\
        \\midrule
{chr(10).join(latex_rows)}
        \\bottomrule
    \\end{{tabular}}%
    }}
    \\label{{tab:t1_network_metrics}}
\\end{{table}}
"""
        print(latex_table)

    def evaluate_tier2(self, country1='Spain', country2='United Kingdom'):
        """
        Evaluates the generated Tier 2 quantum backhaul for two target countries.
        Assuming Tier 1 has already been run prior to calling this.
        """
        target_countries = [country1, country2]

        print(f"\n--- Running Tier 2 Evaluation for {country1} and {country2} ---")

        # 1. Generate Tier 2 for these specific countries
        self.generate_tier2(target_countries=target_countries)

        # 2. Make sure all nodes have a 'country' attribute for filtering.
        # (Tier 1 nodes might have missed it, but they exist in self.cities usually)
        city_to_country = {}
        if hasattr(self, 'cities') and 'city_name' in self.cities.columns and 'country' in self.cities.columns:
            city_to_country = dict(zip(self.cities['city_name'], self.cities['country']))

        for n, d in self.G.nodes(data=True):
            if 'country' not in d:
                d['country'] = city_to_country.get(n, 'Unknown')

        # 3. Create subgraphs purely for internal resource metrics (like CV/DV counts)
        nodes_c1 = [n for n, d in self.G.nodes(data=True) if d.get('country') == country1]
        nodes_c2 = [n for n, d in self.G.nodes(data=True) if d.get('country') == country2]
        nodes_both = nodes_c1 + nodes_c2

        sub_c1 = self.G.subgraph(nodes_c1)
        sub_c2 = self.G.subgraph(nodes_c2)
        sub_both = self.G.subgraph(nodes_both)

        # Helper function to calculate metrics allowing global routing via self.G
        def calc_metrics(H, name, is_country_subgraph=False, highlight_color='#FF00FF'):
            nodes_of_interest = list(H.nodes())
            N = len(nodes_of_interest)

            # Internal infrastructure (edges actually built within this specific subset)
            E = H.number_of_edges()
            dv_edges = sum(1 for u, v, d in H.edges(data=True) if 'DV' in d.get('edge_type', ''))
            cv_edges = sum(1 for u, v, d in H.edges(data=True) if 'CV' in d.get('edge_type', ''))
            avg_k = (2 * E) / N if N > 0 else 0

            # Find how many of these nodes belong to the Main European Grid (LCC of self.G)
            g_components = list(nx.connected_components(self.G))
            best_cc_in_G = set()
            max_h_nodes = 0
            for comp in g_components:
                h_nodes_in_comp = comp.intersection(nodes_of_interest)
                if len(h_nodes_in_comp) > max_h_nodes:
                    max_h_nodes = len(h_nodes_in_comp)
                    best_cc_in_G = comp

            # The nodes that are successfully connected to the main grid
            h_lcc_nodes = list(best_cc_in_G.intersection(nodes_of_interest))
            perc_lcc = (len(h_lcc_nodes) / N) * 100 if N > 0 else 0

            # DIAGNOSTIC VISUALIZATION: Highlight nodes that failed to connect to the main grid
            if is_country_subgraph:
                for n in nodes_of_interest:
                    if n not in h_lcc_nodes:
                        self.G.nodes[n]['color'] = '#FF00FF'  # Magenta for isolated nodes
                        self.G.nodes[n]['type'] = str(self.G.nodes[n].get('type', '')) + ' (ISOLATED)'

            # Calculate Global Shortest Paths and Diameter specifically for the connected nodes in this country
            if len(h_lcc_nodes) > 1:
                total_path_len = 0
                max_len = 0
                start_node, end_node = None, None
                pair_count = 0

                # We calculate shortest paths over the ENTIRE network (self.G),
                # allowing packets to cross borders (like through Portugal) if it's faster!
                for u in h_lcc_nodes:
                    lengths_from_u = nx.single_source_shortest_path_length(self.G, u)
                    for v in h_lcc_nodes:
                        if u != v:
                            d = lengths_from_u.get(v, 0)
                            total_path_len += d
                            pair_count += 1
                            if d > max_len:
                                max_len = d
                                start_node, end_node = u, v

                L = total_path_len / pair_count if pair_count > 0 else 0
                D = max_len

                # DIAGNOSTIC VISUALIZATION: Highlight the diameter path crossing any country
                if is_country_subgraph and start_node and end_node:
                    path = nx.shortest_path(self.G, start_node, end_node)
                    # Repaint these edges in the main graph G
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if self.G.has_edge(u, v):
                            self.G[u][v]['color'] = highlight_color
                            self.G[u][v]['edge_type'] = 'T1_DIAMETER_PATH'
            else:
                L, D = 0, 0

            return f"        {name} & {N} & {dv_edges} & {cv_edges} & {avg_k:.2f} & {L:.2f} & {D} & {perc_lcc:.1f}\\% \\\\"

        rows = [
            calc_metrics(sub_c1, country1, is_country_subgraph=True, highlight_color='#FF1493'),
            # Deep Pink for Country 1
            calc_metrics(sub_c2, country2, is_country_subgraph=True, highlight_color='#00FFFF'),
            # Cyan for Country 2
            calc_metrics(sub_both, f"{country1} + {country2}")
            # Skip highlighting on combined graph so it doesn't overwrite
        ]

        # 4. Generate LaTeX Table
        latex_table = f"""
\\subsubsection{{Complex Network Parameters Comparison (Tier 2)}}
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Quantitative comparison of topological metrics for Tier 2 target countries. ($N$ = Nodes, $L$ = Average Shortest Path Length computed via the global network, $D$ = Diameter via the global network, \\% LCC = Percentage of nodes connected to the main European backbone).}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lrrrrrrr}}
        \\toprule
        Region & $N$ & DV Edges & CV Edges & $\\langle k \\rangle$ & Avg $L$ & Max $L$ ($D$) & \\% LCC \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}%
    }}
    \\label{{tab:t2_network_metrics}}
\\end{{table}}
"""
        print(latex_table)

        # 5. Generate the single Folium Map for the studied area
        if hasattr(self, 'plot_network'):
            map_name = f"tier2_{country1.replace(' ', '')}_{country2.replace(' ', '')}"
            self.plot_network(mode_name=map_name)

    def evaluate_tier3(self, region1='Catalonia', region2='Bavaria'):
        """
        Evaluates the generated Tier 3 quantum backhaul for two target regions.
        Assuming Tier 1 and Tier 2 have already been run prior to calling this.
        """
        print(f"\n--- Running Tier 3 Evaluation for {region1} and {region2} ---")

        # 1. Generate Tier 3 for these specific regions.
        # We call it sequentially to safely bypass the target_regions[0] limit in the generator.
        self.generate_tier3(target_regions=[region1])
        self.generate_tier3(target_regions=[region2])

        # 2. Make sure all nodes have a 'region' attribute for filtering.
        # (Tier 1/2 nodes might have missed it, but they exist in self.cities usually)
        city_to_region = {}
        if hasattr(self, 'cities') and 'city_name' in self.cities.columns and 'region' in self.cities.columns:
            city_to_region = dict(zip(self.cities['city_name'], self.cities['region']))

        for n, d in self.G.nodes(data=True):
            if 'region' not in d:
                d['region'] = city_to_region.get(n, 'Unknown')

        # 3. Create subgraphs purely for internal resource metrics (like CV/DV counts)
        nodes_r1 = [n for n, d in self.G.nodes(data=True) if d.get('region') == region1]
        nodes_r2 = [n for n, d in self.G.nodes(data=True) if d.get('region') == region2]
        nodes_both = nodes_r1 + nodes_r2

        sub_r1 = self.G.subgraph(nodes_r1)
        sub_r2 = self.G.subgraph(nodes_r2)
        sub_both = self.G.subgraph(nodes_both)

        # Helper function to calculate metrics allowing global routing via self.G
        def calc_metrics(H, name, is_region_subgraph=False, highlight_color='#FF00FF'):
            nodes_of_interest = list(H.nodes())
            N = len(nodes_of_interest)

            # Internal infrastructure (edges actually built within this specific subset)
            E = H.number_of_edges()
            dv_edges = sum(1 for u, v, d in H.edges(data=True) if 'DV' in d.get('edge_type', ''))
            cv_edges = sum(1 for u, v, d in H.edges(data=True) if 'CV' in d.get('edge_type', ''))
            avg_k = (2 * E) / N if N > 0 else 0

            # Find how many of these nodes belong to the Main European Grid (LCC of self.G)
            g_components = list(nx.connected_components(self.G))
            best_cc_in_G = set()
            max_h_nodes = 0
            for comp in g_components:
                h_nodes_in_comp = comp.intersection(nodes_of_interest)
                if len(h_nodes_in_comp) > max_h_nodes:
                    max_h_nodes = len(h_nodes_in_comp)
                    best_cc_in_G = comp

            # The nodes that are successfully connected to the main grid
            h_lcc_nodes = list(best_cc_in_G.intersection(nodes_of_interest))
            perc_lcc = (len(h_lcc_nodes) / N) * 100 if N > 0 else 0

            # DIAGNOSTIC VISUALIZATION: Highlight nodes that failed to connect to the main grid
            if is_region_subgraph:
                for n in nodes_of_interest:
                    if n not in h_lcc_nodes:
                        self.G.nodes[n]['color'] = '#FF00FF'  # Magenta for isolated nodes
                        self.G.nodes[n]['type'] = str(self.G.nodes[n].get('type', '')) + ' (ISOLATED)'

            # Calculate Global Shortest Paths and Diameter specifically for the connected nodes in this region
            if len(h_lcc_nodes) > 1:
                total_path_len = 0
                max_len = 0
                start_node, end_node = None, None
                pair_count = 0

                # We calculate shortest paths over the ENTIRE network (self.G)
                for u in h_lcc_nodes:
                    lengths_from_u = nx.single_source_shortest_path_length(self.G, u)
                    for v in h_lcc_nodes:
                        if u != v:
                            d = lengths_from_u.get(v, 0)
                            total_path_len += d
                            pair_count += 1
                            if d > max_len:
                                max_len = d
                                start_node, end_node = u, v

                L = total_path_len / pair_count if pair_count > 0 else 0
                D = max_len

                # DIAGNOSTIC VISUALIZATION: Highlight the diameter path crossing any region
                if is_region_subgraph and start_node and end_node:
                    path = nx.shortest_path(self.G, start_node, end_node)
                    # Repaint these edges in the main graph G
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if self.G.has_edge(u, v):
                            self.G[u][v]['color'] = highlight_color
                            self.G[u][v]['edge_type'] = 'T3_DIAMETER_PATH'
            else:
                L, D = 0, 0

            return f"        {name} & {N} & {dv_edges} & {cv_edges} & {avg_k:.2f} & {L:.2f} & {D} & {perc_lcc:.1f}\\% \\\\"

        rows = [
            calc_metrics(sub_r1, region1, is_region_subgraph=True, highlight_color='#FF1493'),
            # Deep Pink for Region 1
            calc_metrics(sub_r2, region2, is_region_subgraph=True, highlight_color='#00FFFF'),  # Cyan for Region 2
            calc_metrics(sub_both, f"{region1} + {region2}")  # Skip highlighting on combined graph
        ]

        # 4. Generate LaTeX Table
        latex_table = f"""
\\subsubsection{{Complex Network Parameters Comparison (Tier 3)}}
\\begin{{table}}[htbp]
    \\centering
    \\caption{{Quantitative comparison of topological metrics for Tier 3 target regions. ($N$ = Nodes, $L$ = Average Shortest Path Length computed via the global network, $D$ = Diameter via the global network, \\% LCC = Percentage of nodes connected to the main European backbone).}}
    \\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{lrrrrrrr}}
        \\toprule
        Region & $N$ & DV Edges & CV Edges & $\\langle k \\rangle$ & Avg $L$ & Max $L$ ($D$) & \\% LCC \\\\
        \\midrule
{chr(10).join(rows)}
        \\bottomrule
    \\end{{tabular}}%
    }}
    \\label{{tab:t3_network_metrics}}
\\end{{table}}
"""
        print(latex_table)

        # 5. Generate the single Folium Map for the studied area
        if hasattr(self, 'plot_network'):
            map_name = f"tier3_{region1.replace(' ', '')}_{region2.replace(' ', '')}"
            self.plot_network(mode_name=map_name)
# 1. Load your newly generated CSV
df = pd.read_csv('ultimate_city_coordinates.csv')

# Initialize the builder
net3 = QuantumNetworkBuilder(df)

# 1. Continental Core
net3.generate_tier1(mode = 'backbone_spoke')

#net3.evaluate_tier1_modes()

# 2. National Skeleton (Spain and Germany)
net3.generate_tier2(target_countries=['Spain', 'Germany'])
#net3.evaluate_tier2(country2 = 'Germany')
# 3 & 4. Regional and Fractal Networks (Run them ONE BY ONE)
regions_to_map = ['Catalonia', 'Bavaria']

for region in regions_to_map:
    print(f"\n--- INITIATING BUILD FOR: {region.upper()} ---")
    net3.generate_tier3(target_regions=[region])
    net3.generate_tier4_fractal(target_regions=[region], pop_scale=500)
#net3.evaluate_tier3()
# 5. Plot with the Dynamic Edge Colors
net3.plot_network_top()

