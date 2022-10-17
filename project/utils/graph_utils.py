from pathlib import Path
import sys

import torch
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def build_arcs_dataframe(df):
	'''
	returns the dataframe of consecutive arcs
	'''
	dataframe = df.copy()
	dataframe = dataframe.rename({"activity":"source"},axis=1)
	dataframe.drop(["timestamp"], axis=1, inplace=True)
	dataframe["destination"] = dataframe["source"].shift(-1)
	new_dataframe = dataframe[dataframe["source"]!='|'].copy()
	new_dataframe.dropna(inplace=True)
	new_dataframe.reset_index(drop=True, inplace=True)
	new_dataframe["timestamp"] = new_dataframe.index
	return new_dataframe


def build_temp_graph(dataframe):
	activities = set(dataframe["source"].unique()).union({'|'})
	temporal_graph = {}
	for activity in activities:
		df = dataframe[(dataframe["source"] == activity) | (dataframe["destination"] == activity)]
		temporal_neighborhood = []
		for index, row in df.iterrows():
			neighbor = row["source"] if row["source"] != activity else row["destination"]
			temporal_neighborhood.append([index, neighbor])
		temporal_graph[activity] = pd.DataFrame(temporal_neighborhood, columns=["time", "activity"])
	return temporal_graph


def node_temporal_feature(node, temporal_graph, time, window):
	if node in temporal_graph:
		node_temporal_graph = temporal_graph[node]
		condition = (node_temporal_graph["time"] >= time-window) & (node_temporal_graph["time"] <= time)
		valid_rows = node_temporal_graph[condition].sort_values("activity")
		unique_neighbors = valid_rows["activity"].unique()
		signature = []
		for un in unique_neighbors:
			df = valid_rows.copy()
			df = df[df["activity"] == un]
			df["time"] = df["time"] - valid_rows["time"].min()
			s = np.zeros((window+1),float)
			for value in df["time"]:
				s[value] = 1
			signature.append(s)
		features = None
		if (len(signature) > 0):
			features = signature[0]
			for i in range(1, len(signature)):
				features = np.concatenate((features, signature[i]), axis=0)
		if features is not None:
			return features
	return np.zeros((window+1),float)


def get_feature_given_window(dataframe, tg, window):
	f = []
	df = dataframe.copy()
	for index, row in df.iterrows():
		source = node_temporal_feature(row["source"], tg, index, window)
		destination = node_temporal_feature(row["destination"], tg, index, window)
		if (len(source) < len(destination)):
			element_sum = destination.copy()
			element_sum[:len(source)] += source
		else:
			element_sum = source.copy()
			element_sum[:len(destination)] += destination
		f.append(sum(element_sum))
	feature = f"f_{window}"
	if sum(f) > 0:
		df[feature] = f / np.linalg.norm(f)
	df = df.fillna(value=0)
	return df


def generate_features(dataframe, tg, n):
	df = dataframe.copy()
	for i in range(1, n+1):
		df = get_feature_given_window(df, tg, i)
	return df


def temporal_embedding(dataframe, temporal_graph, encoding, window):
	activities = set(dataframe["source"].unique()).union({'|'})
	embeddings = {}

	df = dataframe.copy()
	for time, row in df.iterrows():
		activity = row["source"]
		if activity not in embeddings:
			embeddings[activity] = []
		tg = temporal_graph[activity]
		temporal_neighbors = tg[(tg["time"] > time-window) & (tg["time"] <= time)].copy()
		min_time = temporal_neighbors["time"].min()
		temporal_neighbors["time"] = temporal_neighbors["time"].apply(lambda x : x-min_time)
		embedding = np.zeros((window), float)
		for _, row in temporal_neighbors.iterrows():
			embedding[row["time"]] = encoding[row["activity"]]
		embeddings[activity].append(embedding)

	end_df = dataframe.copy()
	end_df = end_df[end_df["destination"]=='|']
	embeddings['|'] = []
	for time, row in end_df.iterrows():
		tg = temporal_graph[activity]
		temporal_neighbors = tg[(tg["time"] > time-window) & (tg["time"] <= time)].copy()
		min_time = temporal_neighbors["time"].min()
		temporal_neighbors["time"] = temporal_neighbors["time"].apply(lambda x : x-min_time)
		embedding = np.zeros((window), float)
		for _, row in temporal_neighbors.iterrows():
			embedding[row["time"]] = encoding[row["activity"]]
		embeddings['|'].append(embedding)

	features = {}
	for activity in activities:
		if activity not in features:
			features[activity] = np.zeros((window), float)
		for embedding in embeddings[activity]:
			features[activity] += embedding
		assert len(features[activity]) == window

	return features




def build_graph(unique_activities, places, encoding):
	G = nx.DiGraph()

	nodes = []
	place_io = []
	place_in = {}
	place_out = {}

	for name in encoding.keys():
		if name in unique_activities:
			G.add_node(name)
			nodes.append(name)

	for idx, place in enumerate(places):
		tail = []
		head = []
		tmp_tail, tmp_head = place.split("--->")
		for e in tmp_tail.split(" "):
			if e != "":
				tail.append(e)
		for e in tmp_head.split(" "):
			if e != "":
				head.append(e)

		pname = f"place_{idx}"
		if pname not in place_in:
			place_in[pname] = set()
		if pname not in place_out:
			place_out[pname] = set()



		G.add_node(pname)
		if pname not in nodes:
			nodes.append(pname)

		for e in tail:
			G.add_edge(e, pname)
			place_in[pname].add(e)
		for e in head:
			G.add_edge(pname, e)
			place_out[pname].add(e)

	assert len(place_in) > 0 and len(place_in) > 0
	place_io = [place_in, place_out]

	return G, nodes, place_io


def draw_networkx(g):
	pos = nx.spring_layout(g, k=0.3*1/np.sqrt(len(g.nodes())), iterations=20)
	plt.figure(5, figsize=(30, 30))
	nx.draw(g, pos)
	nx.draw_networkx_labels(g, pos=pos)
	plt.show()


def get_backward_star(edge_index, node_idx):
	backward = []
	degree = 0
	for i in range(len(edge_index[0])):
		if node_idx == edge_index[1][i].item():
			backward.append(edge_index[0][i].item())
			degree += 1
	return backward, degree


def get_forward_star(edge_index, node_idx):
	forward = []
	degree = 0
	for i in range(len(edge_index[1])):
		if node_idx == edge_index[0][i].item():
			forward.append(edge_index[1][i].item())
			degree += 1
	return forward, degree


def is_graph_connected(edge_index, mask):
	if edge_index is None:
		return False

	neighbors = {}
	limit = torch.max(edge_index).item()
	for i in range(limit+1):
		neighbors[i] = set()

	for i in range(len(edge_index[0])):
		src = edge_index[0][i].item()
		dst = edge_index[1][i].item()

		neighbors[src].add(dst)

	if len(neighbors) == 0:
		return False

	start = list(neighbors.keys())[0]
	queue = [start]
	visited = set()

	while queue:
		node = queue.pop(0)
		if node not in visited and mask[node]:
			children = sorted(list(neighbors[node]), reverse=True)
			for child in children:
				queue.insert(0, child)
		visited.add(node)

	return len(visited) == len(neighbors)


def get_next_activities(edge_index, idx, mask=None):
	next_activities = set()
	if mask is not None:
		filter = []
		for i in range(len(edge_index[0])):
			src = edge_index[0][i].item()
			dst = edge_index[1][i].item()
			filter.append(mask[src] and mask[dst])

		edge_index = edge_index[:, filter]

	next_places, _ = get_forward_star(edge_index, idx)
	for place in next_places:
		activities, _ = get_forward_star(edge_index, place)
		for activity in activities:
			next_activities.add(activity)
	return next_activities


def check_activity_connection(edge_index, connections, mask):
	for activity, next_activities in connections.items():
		current_next = get_next_activities(edge_index, activity, mask)
		if current_next != next_activities:
			return False
	return True


def node_degree(node, edge_index):
	_, in_deg = get_backward_star(edge_index, node)
	_, out_deg = get_forward_star(edge_index, node)
	return in_deg + out_deg


def mean_node_degree(nodes, edge_index):
	degrees = []
	for node in nodes:
		degree = node_degree(node, edge_index)
		if degree == 0:
			return None

		degrees.append(degree)

	return np.mean(degrees)


def std_node_degree(nodes, edge_index):
	node_degrees = [node_degree(node, edge_index) for node in nodes]
	mean_degree = mean_node_degree(nodes, edge_index)
	std = 0
	if mean_degree is not None:
		ss = [(degree-mean_degree)**2 for degree in node_degrees]
		std = sum(ss)/len(node_degrees)
	return std


def add_silent_transitions(edge_index, mask, nodes, ar):
	places = [idx for idx in range(nodes.index('|')+1, len(nodes))]

	direct_succession = ar["direct_succession"]

	candidate_positions = set()

	following_places = {}
	for place in places:
		following_places[place] = set()
		if mask[place]:
			forward, _ = get_forward_star(edge_index, place)
			for activity in forward:
				next_places, _ = get_forward_star(edge_index, activity)
				for np in next_places:
					if mask[np]:
						following_places[place].add(np)
						candidate_positions.add((place, np))

	no_silent = 0
	new_idx = torch.max(edge_index) + 1
	new_nodes = []
	new_arcs = [[],[]]

	for position in candidate_positions:
		src, dst = position

		backward, _ = get_backward_star(edge_index, src)
		forward, _ = get_forward_star(edge_index, dst)

		for b in backward:
			for f in forward:
				if nodes[b] in direct_succession and nodes[f] in direct_succession[nodes[b]]:
					new_nodes.append("silent_" + str(no_silent))
					new_arcs[0].append(src)
					new_arcs[1].append(new_idx.item())
					new_arcs[0].append(new_idx.item())
					new_arcs[1].append(dst)
					new_idx += 1
					no_silent += 1

	if len(new_nodes) > 0:
		new_edge_index_0 = torch.tensor(new_arcs[0])
		new_edge_index_0 = torch.reshape(new_edge_index_0, (1, new_edge_index_0.shape[0]))
		new_edge_index_1 = torch.tensor(new_arcs[1])
		new_edge_index_1 = torch.reshape(new_edge_index_1, (1, new_edge_index_1.shape[0]))
		new_edge_index = torch.cat((new_edge_index_0, new_edge_index_1), dim=0)
		edge_index = torch.cat((edge_index, new_edge_index), dim=1)

		nodes = nodes + new_nodes

	return edge_index, nodes


def get_activity_order(edge_index, nodes):
	order = []
	queue = [torch.min(edge_index).item()]
	visited = set()

	while queue:
		current = queue.pop(0)
		if current not in visited:
			order.append(current)
			for i in range(len(edge_index[0])):
				if edge_index[0][i].item() == current:
					queue.append(edge_index[1][i].item())
			visited.add(current)

	return [i for i in order if i <= nodes.index('|')]
	

def get_next_node(edge_index, nodes):
	queue = [torch.min(edge_index).item()]
	visited = set()

	followers = {node:set() for node in nodes}

	while queue:
		current = queue.pop(0)
		if current not in visited:
			for i in range(len(edge_index[0])):
				if edge_index[0][i].item() == current:
					queue.append(edge_index[1][i].item())
					followers[nodes[current]].add(nodes[edge_index[1][i].item()])
			visited.add(current)

	return followers