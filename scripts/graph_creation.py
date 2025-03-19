import torch
from torch_geometric.data import Data
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import momepy as mp
from shapely.geometry import Point
from shapely.strtree import STRtree
from collections import Counter
import pickle
import tqdm
sns.set_theme()

def get_city_graph(lat, lon, dist, features):
    g = ox.graph_from_point((lat, lon), dist=dist, network_type='bike', simplify=True, retain_all=False)
    feat_dict = {i: True for i in features}
    amenities = ox.features.features_from_point((lat, lon), tags=feat_dict, dist=dist)
    amenities = amenities[amenities.geometry.notnull()]
    amenities['amenity'] = amenities[features].bfill(axis=1).iloc[:, 0]
    amenities = amenities[amenities['amenity'].notnull()]
    gdf = mp.nx_to_gdf(g, points=False, lines=True, spatial_weights=True).to_crs(epsg=3857)
    return g, gdf, amenities

def find_nearest_edge(linestrings, point, from_node, to_node):
    shortest_distance = float('inf')
    closest_edge, node_pair = None, None
    for linestring, n1, n2 in zip(linestrings, from_node, to_node):
        if linestring:
            distance = linestring.distance(point)
            if distance < shortest_distance:
                shortest_distance, closest_edge, node_pair = distance, linestring, (n1, n2)
    return closest_edge, shortest_distance, node_pair

def assign_counter_data(g, gdf_new, linestrings, from_node, to_node):
    for _, row in tqdm.tqdm(gdf_new.iterrows(), total=len(gdf_new)):
        point = row['geometry']
        _, shortest_distance, node_pair = find_nearest_edge(
            linestrings, point, from_node, to_node
        )
        if node_pair:
            edge_data = g[node_pair[0]][node_pair[1]][0]
            if 'aadt' not in edge_data or edge_data['aadt_dist'] > shortest_distance:
                edge_data.update({
                    'aadt': row['aadt_cykler'],
                    'aadt_dist': shortest_distance
                })
    for _, _, data in g.edges(data=True):
        data.setdefault('aadt', 0)

def create_torch_graph(g):
    edge_list = [
        ((s, t), (v['bc'], int(v['aadt'])))
        for s, t, v in g.edges(data=True)
    ]
    node_to_idx = {
        node: idx
        for idx, node in enumerate(
            set([n for edge in edge_list for n in edge[0]])
        )
    }
    edge_index = torch.tensor(
        [
            [node_to_idx[src], node_to_idx[tgt]]
            for (src, tgt), _ in edge_list
        ],
        dtype=torch.long
    ).t()
    edge_attr = torch.tensor(
        [attr for _, attr in edge_list],
        dtype=torch.float
    )
    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr[:, 0].unsqueeze(1),
        edge_label=edge_attr[:, 1].unsqueeze(1)
    )
    return graph

lat, lon, dist = 55.6867243, 12.5700724, 10000
features = ['amenity', 'shop', 'building', 'highway', 'natural']

# Get city graph and amenities
g, gdf, amenities = get_city_graph(lat, lon, dist, features)

# Load counter data
gdf_new = gpd.read_file('../data/raw/trafiktaelling.json')
gdf_new = gdf_new[gdf_new['aadt_cykler'].notnull()][['id', 'vejnavn', 'geometry', 'aadt_cykler']]

# Assign counter data to graph
linestrings = [data['geometry'] for _, _, data in g.edges(data=True) if 'geometry' in data]
from_node, to_node = zip(*[(s, t) for s, t, _ in g.edges(data=True)])
assign_counter_data(g, gdf_new, linestrings, from_node, to_node)

# Save graph with counter data
with open('../data/graphs/graph_nx.pkl', 'wb') as f:
    pickle.dump(g, f)

# Create torch-geometric graph
torch_graph = create_torch_graph(g)

# Save torch-geometric graph
with open('../data/graphs/graph_tg.pkl', 'wb') as f:
    pickle.dump(torch_graph, f)
