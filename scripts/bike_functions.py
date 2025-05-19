import torch
from torch_geometric.data import Data
import osmnx as ox
import networkx as nx
import numpy as np
import geopandas as gpd
from folium import plugins
from folium.plugins import HeatMap
import momepy as mp 
import seaborn as sns
from shapely.strtree import STRtree
import pickle
from tqdm import tqdm
import os, glob

def get_city_graph(lat, lon, dist, features, expand_features):
    g = ox.graph_from_point((lat, lon), dist=dist, network_type='bike', simplify=True, retain_all=False)
    feat_dict = {i : True for i in features}
    amenities = ox.features.features_from_point((lat, lon), tags=feat_dict, dist=dist)
    amenities = amenities[amenities.geometry.notnull()]
    amenities['new_col'] = np.nan

    for feat in features:
        if feat not in expand_features:
            amenities.loc[amenities[feat].notnull(), 'new_col'] = feat
    
    amenities['amenity'] = amenities['new_col']

    for feat in expand_features:
        amenities['amenity'].fillna(amenities[feat], inplace=True)
    amenities = amenities[amenities['amenity'].notnull()]

    gdf = mp.nx_to_gdf(g, points=False, lines=True, spatial_weights=True).to_crs(epsg=3857)
    gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)
    return g, gdf, amenities

def create_linegraph(g):
    g = nx.Graph(g)
    H = nx.line_graph(g)
    H.add_nodes_from((node, g.edges[node]) for node in H)   
    for s, t in H.edges:
        H.edges[s, t]['weight'] = g.edges[s]['length'] + g.edges[t]['length']
    return H

def calc_bc(graph):
    ebc = dict(nx.all_pairs_dijkstra_path(graph,
                                    weight='weight',
                                    cutoff=1000,))
    bc = {i: 0 for i in graph.nodes}
    for node in tqdm(graph.nodes):
        for path in ebc[node].values():
            for node_visited in set(path):
                bc[node_visited] += 1
    total_nodes = graph.number_of_nodes() ** 2
    return {node: count / total_nodes for node, count in bc.items()}

def load_aadt(filepath, g, gdf):
    nodes, edges = mp.nx_to_gdf(g)
    gdf2 = gpd.GeoDataFrame.from_file(filepath)
    gdf2.set_crs(epsg=4326, inplace=True)
    gdf2 = gdf2.to_crs(epsg=3857)
    gdf2['geometry'] = gdf2['geometry']
    # gdf2 = gdf2[gdf2['geometry'].within(gdf['geometry'])]
    ### export only relevant columns
    gdf_new = gdf2[['id', 'vejnavn', 'geometry', 'aadt_cykler']]
    ### remove null values on aadt_cykler
    gdf_new = gdf_new[gdf_new['aadt_cykler'].notnull()]
    xmin, ymin, xmax, ymax = gdf.total_bounds
    gdf_new = gdf_new.cx[xmin:xmax, ymin:ymax]
    gdf_new.to_crs(epsg=4326, inplace=True)
    return gdf_new

def assign_aadt_to_graph_edges(g, gdf_new, H, aadt_col='aadt_cykler'):
    """
    Assigns AADT values from gdf_new to the nearest edge in graph H based on proximity.

    Parameters:
    - g: networkx graph with edge geometries.
    - gdf_new: GeoDataFrame containing points and AADT values.
    - H: networkx graph where AADT attributes will be assigned.
    - aadt_col: column name in gdf_new containing the AADT values.
    """

    edges_data = list(g.edges(data=True))
    linestrings = [attr['geometry'] if 'geometry' in attr else None for _, _, attr in edges_data]
    from_node = [u for u, _, _ in edges_data]
    to_node = [v for _, v, _ in edges_data]

    tree = STRtree(linestrings)

    for i, row in tqdm(gdf_new.iterrows(), total=len(gdf_new)):
        point = row['geometry']
        if point is None:
            continue

        nearest_edge_idx = tree.nearest(point)
        nearest_edge = linestrings[nearest_edge_idx]
        nearest_edge_distance = nearest_edge.distance(point)

        start_node = from_node[nearest_edge_idx]
        end_node = to_node[nearest_edge_idx]

        # Ensure the edge exists in H
        if (start_node, end_node) not in H.nodes():
            if (end_node, start_node) not in H.nodes():
                continue
            else:
                start_node, end_node = end_node, start_node

        node_pair = (start_node, end_node)
        over_writes = 0
        # Initialize or update AADT attributes if closer
        if 'aadt' not in H.nodes()[node_pair]:
            H.nodes()[node_pair]['aadt'] = row[aadt_col]
            H.nodes()[node_pair]['aadt_distance'] = nearest_edge_distance
        elif H.nodes()[node_pair]['aadt_distance'] > nearest_edge_distance:
            over_writes += 1
            H.nodes()[node_pair]['aadt'] = row[aadt_col]
            H.nodes()[node_pair]['aadt_distance'] = nearest_edge_distance
        print(f"How many times did we overwrite and AADT Value? {over_writes}")
    return H

def clean_and_standardize_node_features(H, remove_fields=None):
    """
    Cleans and standardizes node attributes in a graph.
    Removes specified fields, ensures all attributes are numeric floats, and fills missing features with 0.

    Parameters:
    - H: networkx graph.
    - remove_fields: list of fields to remove (default common OSM fields).
    
    Returns:
    - all_feats: list of all standardized features across nodes.
    """
    if remove_fields is None:
        remove_fields = ['geometry', 'name', 'highway', 'ref', 'aadt_dist', 'aadt_distance']

    # Clean node attributes and convert to floats where possible
    for node, data in H.nodes(data=True):
        for field in remove_fields:
            data.pop(field, None)

        # Convert to float if possible, else remove
        for key in list(data.keys()):
            if not isinstance(data[key], (int, float)):
                try:
                    data[key] = float(data[key])
                except (ValueError, TypeError):
                    data.pop(key, None)

    # Gather all unique features across all nodes
    all_feats = set()
    for _, data in H.nodes(data=True):
        all_feats.update(data.keys())
    all_feats = list(all_feats)

    # Fill missing features with 0
    for _, data in H.nodes(data=True):
        for feat in all_feats:
            data.setdefault(feat, 0)

    return all_feats

def graph_to_linegraph_data(H, all_feats, target_feat='aadt', osmid_feat='osmid'):
    """
    Converts a networkx graph H with node and edge attributes into a PyTorch Geometric Data object.
    
    Parameters:
    - H: networkx graph with node features.
    - all_feats: list of feature names to extract from nodes.
    - target_feat: feature to use as the target variable (default 'aadt').
    - osmid_feat: feature to use as osmid identifier (default 'osmid').
    
    Returns:
    - PyTorch Geometric Data object with node features, targets, osmid, and edge index.
    """
    node_list, x, y, osmid_list = [], [], [], []

    for node, feats in H.nodes(data=True):
        node_list.append(node)
        x.append([feats.get(feat, 0.0) for feat in all_feats if feat not in [target_feat, osmid_feat]])
        y.append(feats[target_feat])
        osmid_list.append(feats[osmid_feat])

    node_idx = {node: idx for idx, node in enumerate(node_list)}

    edge_index = [[node_idx[s], node_idx[t]] for s, t in H.edges()]

    data = Data()
    data.num_nodes = len(node_list)
    data.x = torch.tensor(x, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.float)
    data.osmid = torch.tensor(osmid_list, dtype=torch.long)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data.H = H  # Optional: Attach original H graph if needed

    return data

def save_graph_with_config(
    linegraph, 
    H, 
    node_features, 
    features, 
    expand_features, 
    dist, 
    base_path='../data/graphs'
):
    """
    Save graph data, networkx graph, and node features into a structured folder.
    Uses config files to check for existing setups, creates new folders if needed.

    Parameters:
    - linegraph: PyTorch Geometric Data object.
    - H: NetworkX graph object.
    - node_features: DataFrame of node features.
    - features: list of features.
    - expand_features: list of expanded features.
    - dist: distance parameter.
    - base_path: base path to store the graphs and configs.
    
    Returns:
    - num_folder (str): The assigned folder number where data was saved.
    """
    config_folder = glob.glob(f'{base_path}/configs/*.txt')

    def config_matches(file_path):
        with open(file_path, 'r') as f:
            config = f.readlines()
        config_dict = {}
        for line in config:
            key, value = line.strip().split(':', 1)
            config_dict[key.strip()] = set(value.strip().split()) if key != 'distance' else int(value.strip())

        return (
            config_dict.get('features', set()) == set(features) and
            config_dict.get('expand_features', set()) == set(expand_features) and
            config_dict.get('distance', None) == dist
        )

    # Determine folder number
    if not config_folder:
        print('Creating initial folder structure...')
        num_folder = '1'
    else:
        num_folder = None
        for file in config_folder:
            if config_matches(file):
                num_folder = os.path.splitext(os.path.basename(file))[0]
                break
        if not num_folder:
            num_folder = str(len(config_folder) + 1)

    # Create necessary folders and save configs
    os.makedirs(f'{base_path}/{num_folder}/models', exist_ok=True)
    with open(f'{base_path}/configs/{num_folder}.txt', 'w') as f:
        f.write(f"features: {' '.join(features)}\n")
        f.write(f"expand_features: {' '.join(expand_features)}\n")
        f.write(f"distance: {dist}\n")

    # Save data
    with open(f'{base_path}/{num_folder}/linegraph_tg.pkl', 'wb') as f:
        pickle.dump(linegraph, f)

    with open(f'{base_path}/{num_folder}/linegraph_nx.pkl', 'wb') as f:
        pickle.dump(H, f)

    node_features.to_csv(f'{base_path}/{num_folder}/node_features.csv', index=False)

    print(f"Graph and data saved in folder {num_folder}")
    return num_folder
