import torch
from torch_geometric.data import Data
import osmnx as ox
import networkx as nx
import numpy as np
import geopandas as gpd
import momepy as mp 
import seaborn as sns
from shapely.strtree import STRtree
import pickle
from tqdm import tqdm
import os, glob

sns.set_theme()
import yaml

def load_config(file_path="network-creation.yaml"):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
for key in config:
    if len(config[key]) == 1:
        config[key] = config[key][0]
    globals()[key] = config[key]

print(config['dist'])

slopes = gpd.read_file('../data/edges_slope.gpkg').drop(columns = 
                                                        ['node_start', 'node_end', 'edge_id'])

def get_city_graph(lat, lon, dist, features = ['amenity', 'shop', 'building'], expand_features = ['amenity']):
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

g, gdf, amenities = get_city_graph(lat,
                                    lon,
                                    10000,
                                    features = features)

def create_linegraph(g):
    g = nx.Graph(g)
    H = nx.line_graph(g)
    H.add_nodes_from((node, g.edges[node]) for node in H)   
    for s, t in H.edges:
        H.edges[(s, t)]['weight'] = g.edges[s]['length'] + g.edges[t]['length']
    return H

H = create_linegraph(g)

def calc_bc(shortest_paths, graph):
    bc = {i: 0 for i in graph.nodes}
    for node in tqdm(graph.nodes):
        for path in shortest_paths[node].values():
            for node_visited in set(path):
                bc[node_visited] += 1
    total_nodes = graph.number_of_nodes() ** 2
    return {node: count / total_nodes for node, count in bc.items()}

ebc = dict(nx.all_pairs_dijkstra_path(H, weight='weight',
                                        cutoff=100,
                                        ))
bc = calc_bc(ebc, H)

bc2 = {}
for x, y in bc:
    bc2[(x, y)] = bc[(x, y)]

nx.set_node_attributes(H, bc2, 'bc')

nodes, edges = mp.nx_to_gdf(g)
filepath = '../data/raw/trafiktaelling.json'
gdf = gpd.GeoDataFrame.from_file(filepath)
gdf.set_crs(epsg=4326, inplace=True)
gdf['geometry'] = gdf['geometry']
### export only relevant columns
gdf_new = gdf[['id', 'vejnavn', 'geometry', 'aadt_cykler']]
### remove null values on aadt_cykler
gdf_new = gdf_new[gdf_new['aadt_cykler'].notnull()]

linestrings = [i[2]['geometry'] if 'geometry' in i[2] else None for i in list(g.edges(data=True))]
from_node = [i[0] for i in list(g.edges(data=True))]
to_node = [i[1] for i in list(g.edges(data=True))]

tree = STRtree(linestrings)
for i, row in tqdm(gdf_new.iterrows(), total=len(gdf_new)):
    point = row['geometry']
    if point is None:
        continue
    nearest_edge_idx = tree.nearest(point)
    nearest_edge = linestrings[nearest_edge_idx]
    nearest_edge_distance = nearest_edge.distance(point)
    start_node = from_node[linestrings.index(nearest_edge)]
    end_node = to_node[linestrings.index(nearest_edge)]
    
    # Ensure the edge exists in the graph
    if (start_node, end_node) not in H.nodes():
        if (end_node, start_node) not in H.nodes:
            continue
        else:
            start_node, end_node = end_node, start_node

    if 'aadt' not in H.nodes()[(start_node, end_node)]:
        H.nodes()[(start_node, end_node)]['aadt'] = row['aadt_cykler']
        H.nodes()[(start_node, end_node)]['aadt_distance'] = nearest_edge_distance
    if 'aadt_distance' not in H.nodes()[(start_node, end_node)] or H.nodes()[(start_node, end_node)]['aadt_distance'] > nearest_edge_distance:
        H.nodes()[(start_node, end_node)]['aadt'] = row['aadt_cykler']
        H.nodes()[(start_node, end_node)]['aadt_distance'] = nearest_edge_distance

for node, value in H.nodes(data=True):
    if 'aadt' not in value.keys():
        value['aadt'] = 0

# amenities = amenities.reset_index()
nodes = list((node, linestring) for node, linestring in H.nodes(data='geometry'))
nodes = [node for node in nodes if node[1] is not None]
linestrings = [linestring for node, linestring in nodes]
nodes = [node for node, linestring in nodes]
assert len(nodes) == len(linestrings)
amenities['geometry'] = amenities['geometry'].apply(lambda x: x.centroid if x.geom_type == 'Polygon' else x)
tree = STRtree(linestrings)
for geom, amenity in zip(amenities['geometry'], amenities['amenity']):
    nearest = tree.nearest(geom)
    nearest = nodes[nearest]
    if 'amenity' not in H.nodes[nearest]:
        H.nodes[nearest]['amenity'] = [amenity]
    else:
        H.nodes[nearest]['amenity'].append(amenity)

from collections import Counter
for i in H.nodes(data=True):
    if 'amenity' in i[1]:
        amenity_counts = Counter(i[1]['amenity'])
        for key in amenity_counts:
            H.nodes[i[0]][key] = amenity_counts[key]
        ## drop the amenity key
        H.nodes[i[0]].pop('amenity', None)

for node in H.nodes(data=True):
    node[1].pop('geometry', None)
    node[1].pop('osmid', None)
    node[1].pop('name', None)
    node[1].pop('highway', None)
    node[1].pop('ref', None)
    node[1].pop('aadt_dist', None)
    for key in list(node[1].keys()):
        if type(node[1][key]) not in (int, float):
            try:
                node[1][key] = float(node[1][key])
            except:
                node[1].pop(key, None)


all_feats = []
for node in H.nodes(data=True):
    for key in node[1].keys():
        if key not in all_feats:
            all_feats.append(key)

for node in H.nodes(data=True):
    for feat in all_feats:
        if feat not in node[1].keys():
            node[1][feat] = 0

node_list, x, y = [], [], []
for node, feats in list(H.nodes(data=True)):
    node_list.append(node)
    x.append([feats[feat] for feat in all_feats if feat != 'aadt'])
    y.append(feats['aadt'])

node_idx = {node : idx for idx, node in enumerate(node_list)}
edge_index = []
for s, t in list(H.edges):
    edge_index.append([node_idx[s], node_idx[t]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t()
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

linegraph = Data()
linegraph.num_nodes = len(node_list)
linegraph.x = x
linegraph.y = y
linegraph.edge_index = edge_index

# ### create file structure based on number of node features
# config_folder = glob.glob('../data/graphs/configs/*.txt')
# num_folder = None

# if len(config_folder) == 0:
#     print('Creating new folder')
#     os.mkdir('../data/graphs/1')
#     os.mkdir('../data/graphs/1/models')
#     with open('../data/graphs/configs/1.txt', 'w') as f:
#         f.write(f'features: {' '.join(features)}\n')
#         f.write(f'expand_features: {' '.join(expand_features)}\n')
#         f.write(f'distance: {dist}\n')

# for file in config_folder:
#     same_features = False
#     same_expand_features = False
#     same_distance = False
#     with open(file, 'r') as f:
#         config = f.readlines()
#     for line in config:
#         if 'distance' in line:
#             if int(line.split(':')[-1]) == dist:
#                 same_distance = True
#         if 'features' in line:
#             if set(line.split(':')[-1].split()) == set(features):
#                 same_features = True
#         if 'expand_features' in line:
#             if set(line.split(':')[-1].split()) == set(expand_features):
#                 same_expand_features = True
#     if same_distance and same_features and same_expand_features:
#         num_folder = file.split('/')[-1].split('.')[0]
#     else:
#         num_folder = len(config_folder) + 1
#         os.mkdir(f'../data/graphs/{num_folder}')
#         os.mkdir(f'../data/graphs/{num_folder}/models')
#         with open(f'../data/graphs/configs/{num_folder}.txt', 'w') as f:
#             f.write(f'features: {' '.join(features)}\n')
#             f.write(f'expand_features: {' '.join(expand_features)}\n')
#             f.write(f'distance: {dist}\n')
#     with open(f'../data/graphs/{num_folder}/linegraph_tg.pkl', 'wb') as f:
#         pickle.dump(linegraph, f)

# for file in config_folder:
#     same_features = False
#     same_expand_features = False
#     same_distance = False
#     with open(file, 'r') as f:
#         config = f.readlines()
#     for line in config:
#         if 'distance' in line:
#             if int(line.split(':')[-1]) == dist:
#                 same_distance = True
#         if 'features' in line:
#             if set(line.split(':')[-1].split()) == set(features):
#                 same_features = True
#         if 'expand_features' in line:
#             if set(line.split(':')[-1].split()) == set(expand_features):
#                 same_expand_features = True
#     if same_distance and same_features and same_expand_features:
#         num_folder = file.split('/')[-1].split('.')[0]
#     else:
#         num_folder = len(config_folder) + 1
#         os.mkdir(f'../data/graphs/{num_folder}')
#         os.mkdir(f'../data/graphs/{num_folder}/models')
#         with open(f'../data/graphs/configs/{num_folder}.txt', 'w') as f:
#             f.write(f'features: {' '.join(features)}\n')
#             f.write(f'expand_features: {' '.join(expand_features)}\n')
#             f.write(f'distance: {dist}\n')
#     with open(f'../data/graphs/{num_folder}/linegraph_tg.pkl', 'wb') as f:
#         pickle.dump(linegraph, f)


config_folder = glob.glob('../data/graphs/configs/*.txt')
num_folder = None

# If no config exists, create a new one
if not config_folder:
    print('Creating new folder')
    num_folder = 1
    os.makedirs(f'../data/graphs/{num_folder}/models', exist_ok=True)
    with open(f'../data/graphs/configs/{num_folder}.txt', 'w') as f:
        f.write(f'features: {' '.join(features)}\n')
        f.write(f'expand_features: {' '.join(expand_features)}\n')
        f.write(f'distance: {dist}\n')
else:
    for file in config_folder:
        same_features = same_expand_features = same_distance = False
        
        with open(file, 'r') as f:
            config = f.readlines()
        
        for line in config:
            key, value = line.strip().split(": ", 1)
            value = set(value.split())

            if key == "distance" and int(value.pop()) == dist:
                same_distance = True
            elif key == "features" and value == set(features):
                same_features = True
            elif key == "expand_features" and value == set(expand_features):
                same_expand_features = True

        # If a matching config is found, use its folder number
        if same_distance and same_features and same_expand_features:
            num_folder = file.split('/')[-1].split('.')[0]
            break
    
    # If no matching config was found, create a new folder
    if num_folder is None:
        num_folder = len(config_folder) + 1
        os.makedirs(f'../data/graphs/{num_folder}/models', exist_ok=True)
        with open(f'../data/graphs/configs/{num_folder}.txt', 'w') as f:
            f.write(f'features: {' '.join(features)}\n')
            f.write(f'expand_features: {' '.join(expand_features)}\n')
            f.write(f'distance: {dist}\n')
linegraph.graph = num_folder

# Save the line graph in the determined folder
with open(f'../data/graphs/{num_folder}/linegraph_tg.pkl', 'wb') as f:
    pickle.dump(linegraph, f)
