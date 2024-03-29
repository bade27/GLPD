from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import pm4py
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import networkx as nx
from utils.general_utils import create_dirs, dump_to_pickle, create_dirs, split_move_files
from utils.petri_net_utils import add_places, get_alpha_relations, build_net_from_places, reduce_silent_transitions
from utils.pm4py_utils import display_petri_net, generate_trees, get_variants_parsed, log_to_dataframe, save_log_xes, save_petri_net_to_img, save_petri_net_to_pnml, load_log_xes, behavior
from utils.graph_utils import build_arcs_dataframe, build_graph, build_temp_graph, generate_features, temporal_embedding, draw_networkx, get_backward_star, get_forward_star, mean_node_degree, std_node_degree, get_activity_order, get_next_nodes, get_prev_nodes
from sklearn.cluster import KMeans
import model.structure as model_structure
import shutil
from sklearn.preprocessing import KBinsDiscretizer
from data_handling import stats as st


class Dataset():
	def __init__(self, data_dir, type_of_features="temporal", synth=True, filename=""):
		self.data_dir = data_dir
		self.synth = synth

		self.seed = 1234

		self.graphs_dir = os.path.join(self.data_dir, "graphs")
		self.logs_dir = os.path.join(self.graphs_dir, "logs")
		self.raw_dir = os.path.join(self.graphs_dir, "raw")
		self.graph_nodes_dir = os.path.join(self.graphs_dir, "nodes")
		self.graph_variants_dir = os.path.join(self.graphs_dir, "variants")
		self.graph_order_dir = os.path.join(self.graphs_dir, "order")
		self.graph_next_dir = os.path.join(self.graphs_dir, "next")
		self.graph_prev_dir = os.path.join(self.graphs_dir, "prev")
		self.saved_dir = os.path.join(self.data_dir, "saved_images_pnml")
		self.networkx_dir = os.path.join(self.data_dir, 'networkx')
		self.alpha_relations_dir = os.path.join(self.graphs_dir, 'alpha_relations')

		self.input_dir = os.path.join(self.saved_dir, "input_nets")
		self.input_dir_imgs = os.path.join(self.input_dir, "images")
		self.input_dir_pnml = os.path.join(self.input_dir, "pnml")

		self.type_of_features = type_of_features

		self.dirs = [
			self.data_dir, self.graphs_dir, 
			self.logs_dir, self.raw_dir, self.graph_nodes_dir, self.graph_variants_dir,
			self.networkx_dir, self.alpha_relations_dir, self.saved_dir,
			self.input_dir, self.input_dir_imgs, self.input_dir_pnml,
			self.graph_order_dir, self.graph_next_dir, self.graph_prev_dir
		]

		if self.synth:
			self.original_dir = os.path.join(self.saved_dir, "original_nets")
			self.original_dir_imgs = os.path.join(self.original_dir, "images")
			self.original_dir_pnml = os.path.join(self.original_dir, "pnml")

			# self.reconstr_dir = os.path.join(self.saved_dir, "reconstructed_nets")
			# self.reconstr_dir_imgs = os.path.join(self.reconstr_dir, "images")
			# self.reconstr_dir_pnml = os.path.join(self.reconstr_dir, "pnml")


			self.dirs += [
				self.original_dir, self.original_dir_imgs, self.original_dir_pnml
				# self.reconstr_dir, self.reconstr_dir_imgs, self.reconstr_dir_pnml
				]
		
		else:
			log_filename = os.path.join(self.data_dir, filename)
			print(log_filename)
			self.log = load_log_xes(log_filename)

		create_dirs(self.dirs)
		self.columns = ["source", "destination", "timestamp", "case", "log", "state_label"]

		self.set_statistics()


	def get_encoding(self):
		encoding = {}
		encoding['>'] = 0
		if self.synth:
			for k in range(97,123):
				encoding[chr(k)] = k-97+1
			encoding['|'] = 27
		else:
			unique_activities = set()
			for trace in self.log:
				for activity in trace:
					unique_activities.add(activity["concept:name"].replace(" ", "_"))
			count = 1
			for unique in unique_activities:
				encoding[unique] = count
				count += 1
			encoding['|'] = count

		return encoding


	def set_statistics(
		self, mode=[8], min=[4], max=[15], sequence=[0.4], choice=[0.32], parallel=[0.2], 
		loop=[0.0], or_gate=[0.0], silent=[0.0], no_models=1000, no_traces=10, no_datasets=1, no_features=10):

		statistics = {
			'mode': mode, 
			'min':min, 
			'max':max,
			'sequence':sequence,
			'choice':choice,
			'parallel':parallel,
			'loop':loop,
			'or':or_gate,
			'silent':silent,
			'no_models': no_models,
			'no_traces': no_traces,
			'no_datasets': no_datasets,
			'no_features': no_features
		}

		for key, value in statistics.items():
			if key not in ['no_models', 'no_traces', 'no_datasets', 'no_features']:
				assert len(value) == no_datasets

		self.statistics = statistics


	def generate_dataset(self, save_networkx = False, save_images = False, save_models = False, visualize_nets = False, redundancy=1, depth=1, k=30):

		encoding = self.get_encoding()

		dataset_l = []

		if self.synth:
			no_datasets = self.statistics['no_datasets']
			pad_len = self.statistics["no_models"]*self.statistics['no_datasets']
		else:
			no_datasets = 1
			pad_len = 1

		no_features = self.statistics['no_features']

		pad = len(str(pad_len))

		total_csv_entries = 0
		number_of_models = 0
	
		for c in range(no_datasets):
			if self.synth:
				parameters = {k:v[c] for k, v in self.statistics.items() if k not in ['no_models', 'no_traces', 'no_datasets', 'no_features']}
				parameters.update({'no_models': self.statistics['no_models']*redundancy})

				no_traces = self.statistics['no_traces']

				trees = generate_trees(parameters)

				no_models = parameters["no_models"]
			else:
				no_models = 1

			data = pd.DataFrame()
	
			for i in tqdm(range(no_models)):
				if self.synth:
					# log and initial df
					tree = trees[i] if parameters["no_models"] > 1 else trees
					net_ws, im_ws, fm_ws = pm4py.convert_to_petri_net(tree)

					net, im, fm = reduce_silent_transitions(net_ws, im_ws, fm_ws)
					log = pm4py.play_out(
                        net, im, fm, parameters={"no_traces":no_traces})
					print(f"\noriginal number of traces {len(log)}")
					# net, im, fm = pm4py.discover_petri_net_alpha(log)
					# if not is_sound(net, im, fm):
					#     continue
					# log = pm4py.play_out(net, im, fm, parameters={"no_traces":no_traces})
				else:
					log = self.log
					print(f"\noriginal number of traces {len(log)}")
					df_from_log = pm4py.convert_to_dataframe(log)
					df_frequent_act = df_from_log["concept:name"].value_counts()
					top_18 = df_frequent_act.sort_values(ascending=False)[:18]
					df_from_log = df_from_log[df_from_log["concept:name"].isin(top_18.index)]
					assert 0 < len(df_from_log["concept:name"].unique()) <= 18
					log = pm4py.convert_to_event_log(df_from_log)
				
				top_variants, _ = behavior(log, 0.8)
				top = len(top_variants)
				if top < 30:
					top = 30
				elif top > 75:
					top = 75
				filtered_log = pm4py.filter_variants_top_k(log, top)
				log = filtered_log
				print(f"filtered number of traces {len(log)}, that represent the top {top} traces")
				
				unique_activities = set()

				for trace in log:
					for activity in trace:
						unique_activities.add(activity["concept:name"].replace(" ", "_"))
			
				df = log_to_dataframe(log)
		
				# unique_activities = set([transition.name for transition in net.transitions])

				# insert special activities '>' (start) and '|' (finish)
				print("inserting special activities...", end=' ')
				unique_cases = list(df["case"].unique())
				s = df.groupby('case')
				df_start_finish = pd.concat(
					[pd.concat([
						pd.DataFrame({'activity': [">"], "timestamp":[0], "case":[unique_cases[k]]}),
						i,
						pd.DataFrame({'activity': ["|"], "timestamp":[0], "case":[unique_cases[k]]})]
					) for k, i in s])
				df_start_finish.reset_index(inplace=True, drop=True)
				print("done")

				# for case in df["case"].unique():
				# 	current_case = df[df["case"]==case].copy()
				# 	df_start = pd.DataFrame({"activity":'>', "timestamp":0, "case":case},index=[0])
				# 	df_finish = pd.DataFrame({"activity":'|', "timestamp":0, "case":case},index=[0])
				# 	df_start_finish = pd.concat([df_start_finish, df_start], axis=0)
				# 	df_start_finish = pd.concat([df_start_finish, current_case], axis=0)
				# 	df_start_finish = pd.concat([df_start_finish, df_finish], axis=0)
				# 	df_start_finish.reset_index(drop=True,inplace=True)

				unique_activities.add('>')
				unique_activities.add('|')

				# new dataframe to manipulate
				new_df = build_arcs_dataframe(df_start_finish)

				# alpha relations
				print("getting alpha relations...", end=' ')
				dict_alpha_relations = get_alpha_relations(log, depth=depth)
				print("done")

				# net_places = find_actual_places(net) # places of the original net

				print("adding places...", end=' ')
				places = add_places(dict_alpha_relations, further_than_one_hop=False) # places of the input net
				print("done")

				# input net
				input_net, input_im, input_fm = build_net_from_places(unique_activities, places)

				# networkx graph of the new net
				graph, nodes, _ = build_graph(unique_activities, places, encoding)

				adj = nx.to_numpy_matrix(graph)

				ones = np.argwhere(adj == 1)

				edge_index_0 = []
				edge_index_1 = []
		
				for one in ones:
					edge_index_0.append(one[0])
					edge_index_1.append(one[1])

				assert len(edge_index_0) == len(edge_index_1)

				original_edge_index = torch.stack(
					(torch.tensor(edge_index_0, dtype=torch.long), torch.tensor(edge_index_1, dtype=torch.long)), dim=0)

				for one in ones:
					edge_index_0.append(one[1])
					edge_index_1.append(one[0])

				assert len(edge_index_0) == len(edge_index_1)

				edge_index = torch.stack(
					(torch.tensor(edge_index_0, dtype=torch.long), torch.tensor(edge_index_1, dtype=torch.long)), dim=0)

				assert len(edge_index[0]) == len(edge_index[1])

				# temporal graph
				print("building temporal graph...", end=' ')
				temporal_graph = build_temp_graph(new_df)
				new_df["log"] = number_of_models
				new_df["state_label"] = [0 for _ in range(len(new_df))]
				df_correct_order = new_df[self.columns]
				
				if self.type_of_features == "tgn":
					new_df_features = generate_features(df_correct_order, temporal_graph, model_structure.window_size)
				else:
					new_df_features = generate_features(df_correct_order, temporal_graph)

				data = pd.concat([data, new_df_features], axis=0)
				data.reset_index(drop=True, inplace=True)
				print("done")
				
				if visualize_nets:
					draw_networkx(graph)
					display_petri_net(input_net, input_im, input_fm)
					if self.synth:
						display_petri_net(net, im, fm)
					# display_petri_net(rec_net, rec_im, rec_fm)

				if save_images:
					save_petri_net_to_img(input_net, input_im, input_fm, os.path.join(self.input_dir, "images", "input_" + str(number_of_models).zfill(pad) + '.png'))
					if self.synth:
						save_petri_net_to_img(net, im, fm, os.path.join(self.original_dir, "images", "original_" + str(number_of_models).zfill(pad) + '.png'))
					# save_petri_net_to_img(rec_net, rec_im, rec_fm, os.path.join(self.reconstr_dir, "images", "rec_" + str(number_of_models).zfill(pad) + '.png'))

				if save_models:
					save_petri_net_to_pnml(input_net, input_im, input_fm, os.path.join(self.input_dir, "pnml", "input_" + str(number_of_models).zfill(pad) + '.pnml'))
					if self.synth:
						save_petri_net_to_pnml(net, im, fm, os.path.join(self.original_dir, "pnml", "original_" + str(number_of_models).zfill(pad) + '.pnml'))
					# save_petri_net_to_pnml(rec_net, rec_im, rec_fm, os.path.join(self.reconstr_dir, "pnml", "rec_" + str(number_of_models).zfill(pad) + '.pnml'))

				if save_networkx:
					dump_to_pickle(os.path.join(self.networkx_dir, 'gx_' + str(number_of_models).zfill(pad)), graph)

				print("prasing variants...", end=' ')
				variants = get_variants_parsed(log, k)
				assert len(variants) == k
				print("done")

				print("getting activity order...",end=' ')
				activity_order = get_activity_order(original_edge_index, nodes)
				print("done")
				print("next and prev nodes...", end=' ')
				next_nodes = get_next_nodes(original_edge_index, nodes)
				prev_nodes = get_prev_nodes(original_edge_index, nodes)
				print("done")

				# save files
				torch.save(edge_index, os.path.join(self.raw_dir, "graph_" + str(number_of_models).zfill(pad) + ".pt"))
				torch.save(original_edge_index, os.path.join(self.raw_dir, "original_" + str(number_of_models).zfill(pad) + ".pt"))

				if self.type_of_features == "random" or self.type_of_features == "temporal":
					h = torch.max(edge_index)+1
					no_activities = nodes.index('|')+1
					no_places = h.item() - no_activities
					
					if self.type_of_features == "random":
						w = model_structure.num_node_features
						x_activities = torch.randn((no_activities, w))
						x_activities = torch.nn.functional.normalize(x_activities, p=2.0, dim=1)

						x_places = torch.zeros((no_places, w))

						x = torch.cat((x_activities, x_places), dim=0)

						assert x.shape[0] == h
						assert x.shape[1] == w
					else:
						w = model_structure.window_size
						print("getting temporal embeddings...", end=' ')
						features = temporal_embedding(new_df, temporal_graph, encoding, w)
						x_list = [None for _ in range(len(nodes))]
						for position, node in enumerate(nodes):
							if 'p' not in node:
								x_list[position] = torch.reshape(torch.from_numpy(features[node]), (1, w))
								x_list[position] = torch.nn.functional.normalize(x_list[position], p=2.0, dim=1)
							else:
								x_list[position] = torch.zeros((1, w))
						x = torch.cat(x_list, dim=0)

						assert x.shape[0] == h
						assert x.shape[1] == model_structure.window_size
						print("done")
	
					torch.save(x, os.path.join(self.raw_dir, "x_" + str(number_of_models).zfill(pad) + ".pt"))


				print("saving files...", end=' ')
				nodes_file_name = os.path.join(self.graph_nodes_dir, "nodes_" + str(number_of_models).zfill(pad))
				dump_to_pickle(nodes_file_name, nodes)

				variants_file_name = os.path.join(self.graph_variants_dir, "variants_" + str(number_of_models).zfill(pad))
				dump_to_pickle(variants_file_name, variants)

				alpha_relations_file_name = os.path.join(self.alpha_relations_dir, "ar_" + str(number_of_models).zfill(pad))
				dump_to_pickle(alpha_relations_file_name, dict_alpha_relations)

				activity_order_file_name = os.path.join(self.graph_order_dir, "order_" + str(number_of_models).zfill(pad))
				dump_to_pickle(activity_order_file_name, activity_order)

				next_nodes_file_name = os.path.join(self.graph_next_dir, "next_" + str(number_of_models).zfill(pad))
				dump_to_pickle(next_nodes_file_name, next_nodes)

				prev_nodes_file_name = os.path.join(self.graph_prev_dir, "prev_" + str(number_of_models).zfill(pad))
				dump_to_pickle(prev_nodes_file_name, prev_nodes)

				save_log_xes(log, os.path.join(self.logs_dir, "log_" + str(number_of_models).zfill(pad) + '.xes'))

				number_of_models += 1
				print("done")

			# rename activities
			data["source"] = data["source"].map(encoding)
			data["destination"] = data["destination"].map(encoding)

			data["destination"] = data["destination"].astype("int")
			data["timestamp"] = [i for i in range(len(data))]

			dataset_l.append(data)
			total_csv_entries += len(data.index)

		total_csv_entries_ctrl = 0
		for d in dataset_l:
			total_csv_entries_ctrl += len(d.index)

		assert total_csv_entries == total_csv_entries_ctrl

		dataset = pd.DataFrame()
		for data in dataset_l:
			dataset = pd.concat([dataset, data], axis=0)
			dataset.reset_index(drop=True, inplace=True)

		assert len(dataset.index) == total_csv_entries

		if self.type_of_features == "random" or self.type_of_features == "temporal":
			assert len(os.listdir(self.logs_dir)) * 3 == len(os.listdir(self.raw_dir))

		if self.synth:
			print(f"{number_of_models}/{self.statistics['no_models']*self.statistics['no_datasets']*redundancy}")
			dataset.to_csv(os.path.join(self.data_dir, 'tmp_data.csv'), index=False)
		else:
			dataset.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)


	def detect_ill_formed(self):
		ill_formed = {}
		pos = 0
		for graph in sorted(os.listdir(self.raw_dir)):
			if 'original_' in graph:
				edge_index = torch.load(os.path.join(self.raw_dir, graph))
				nodes = [i for i in range(torch.max(edge_index)+1)]
				source = 0
				destination = 0
				for node in nodes:
					_, ind = get_backward_star(edge_index, node)
					_, oud = get_forward_star(edge_index, node)
					source += int(ind==0)
					destination += int(oud==0)
				
				ill_formed[pos] = source > 1 or destination > 1
				pos += 1

		assert pos == len([g for g in os.listdir(self.raw_dir) if 'original_' in g])
		return ill_formed


	def build_graph_stats_df(self, ill_formed):
		logs = []
		mean_degrees = []
		std_degrees = []
		num_nodes = []
		removed = []
		idx = 0
		for file in sorted(os.listdir(self.raw_dir)):
			if 'graph_' in file:
				if not ill_formed[idx]:
					edge_index = torch.load(os.path.join(self.raw_dir, file))
					nodes = [i for i in range(torch.max(edge_index)+1)]
					mean_degree = mean_node_degree(nodes, edge_index)
					std_degree = std_node_degree(nodes, edge_index)
					if mean_degree is not None:
						logs.append(idx)
						mean_degrees.append(mean_degree)
						std_degrees.append(std_degree)
						num_nodes.append(len(nodes))
					else:
						removed.append(idx)
				else:
					removed.append(idx)
				idx += 1

		df = pd.DataFrame({'log':logs, 'mean_deg':mean_degrees, 'std_deg':std_degrees, 'num_nodes':num_nodes})
		return df, removed


	def sample(self, dataset, graphs_stat, removed, perc=0.8):
		graphs_stat = graphs_stat.drop(removed)

		discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
		# X = graphs_stat[['mean_deg', 'std_deg']].to_numpy()
		# X_disc = discretizer.fit_transform(X)
		# graphs_stat['mean_deg'] = X_disc[:,0]
		# graphs_stat['sdt_deg'] = X_disc[:,1]

		X = graphs_stat[['mean_deg', 'std_deg', 'num_nodes']].to_numpy()
		kmeans = KMeans(n_clusters=5, random_state=self.seed).fit(X)

		graphs_stat['cluster'] = kmeans.labels_

		graphs_stat = graphs_stat.sample(frac=1, random_state=self.seed)
		graphs_stat.reset_index(drop=True, inplace=True)

		train_dataset = graphs_stat.groupby(['cluster'], group_keys=False).apply(lambda x: x.sample(frac=perc))

		# df_all = graphs_stat.merge(train_dataset.drop_duplicates(), on='log', how='left', indicator=True)

		# condition = df_all['_merge'] == 'left_only'
		# valid_dataset = graphs_stat[condition].groupby('bin', group_keys=False).apply(lambda x: x.sample(frac=0.6))

		# test_dataset = graphs_stat.loc[~graphs_stat['log'].isin((set(train_dataset['log'].unique())).union(set(valid_dataset['log'].unique())))]

		# assert len(graphs_stat) == len(train_dataset) + len(valid_dataset) + len(test_dataset)

		test_dataset = graphs_stat.loc[~graphs_stat['log'].isin(train_dataset['log'].unique())]

		assert len(graphs_stat) == len(train_dataset) + len(test_dataset)

		offset = 0

		train_df = dataset.merge(train_dataset, on='log', how='left')
		train_df = train_df[~train_df['mean_deg'].isna()]
		train_df.reset_index(drop=True, inplace=True)
		train_df['timestamp'] = [i+offset for i in range(len(train_df.index))]

		offset += len(train_df.index)

		# valid_df = dataset.merge(valid_dataset, on='log', how='left')
		# valid_df = valid_df[~valid_df['mean_deg'].isna()]
		# valid_df.reset_index(drop=True, inplace=True)
		# valid_df['timestamp'] = [i+offset for i in range(len(valid_df.index))]

		# offset += len(valid_df.index)

		test_df = dataset.merge(test_dataset, on='log', how='left')
		test_df = test_df[~test_df['mean_deg'].isna()]
		test_df.reset_index(drop=True, inplace=True)
		test_df['timestamp'] = [i+offset for i in range(len(test_df.index))]

		# return train_df, valid_df, test_df
		return train_df, test_df


	def split(self, train_df, test_df):
		# train dirs
		train_graphs_dir = os.path.join(self.data_dir, "train_graphs")
		train_logs_dir = os.path.join(train_graphs_dir, "logs")
		train_nodes_dir = os.path.join(train_graphs_dir, "nodes")
		train_raw_dir = os.path.join(train_graphs_dir, "raw")
		train_variants_dir = os.path.join(train_graphs_dir, "variants")
		train_alpha_relations_dir = os.path.join(train_graphs_dir, "alpha_relations")
		train_order_dir = os.path.join(train_graphs_dir, "order")
		train_next_dir = os.path.join(train_graphs_dir, "next")
		train_prev_dir = os.path.join(train_graphs_dir, "prev")

		train_saved_dir = os.path.join(train_graphs_dir, "saved_images_pnml")
		train_input_dir = os.path.join(train_saved_dir, "input_net")
		train_input_dir_imgs = os.path.join(train_input_dir, "images")
		train_input_dir_pnml = os.path.join(train_input_dir, "pnml")

		train_original_dir = os.path.join(train_saved_dir, "original_net")
		train_original_dir_imgs = os.path.join(train_original_dir, "images")
		train_original_dir_pnml = os.path.join(train_original_dir, "pnml")

		# train_reconstructed_dir = os.path.join(train_saved_dir, "reconstructed_net")
		# train_reconstructed_dir_imgs = os.path.join(train_reconstructed_dir, "images")
		# train_reconstructed_dir_pnml = os.path.join(train_reconstructed_dir, "pnml")

		train_dirs = [
			train_graphs_dir, train_logs_dir, train_nodes_dir, train_raw_dir, train_variants_dir,
			train_alpha_relations_dir, train_saved_dir, train_input_dir, train_input_dir_imgs, 
			train_input_dir_pnml, train_original_dir, train_original_dir_imgs, train_original_dir_pnml,
			train_order_dir, train_next_dir, train_prev_dir]
			# train_reconstructed_dir, train_reconstructed_dir_imgs, train_reconstructed_dir_pnml]


		# # val dirs
		# validation_graphs_dir = os.path.join(self.data_dir, "validation_graphs")
		# validation_logs_dir = os.path.join(validation_graphs_dir, "logs")
		# validation_nodes_dir = os.path.join(validation_graphs_dir, "nodes")
		# validation_raw_dir = os.path.join(validation_graphs_dir, "raw")
		# validation_variants_dir = os.path.join(validation_graphs_dir, "variants")
		# validation_alpha_relations_dir = os.path.join(validation_graphs_dir, "alpha_relations")

		# validation_saved_dir = os.path.join(validation_graphs_dir, "saved_images_pnml")
		# validation_input_dir = os.path.join(validation_saved_dir, "input_net")
		# validation_input_dir_imgs = os.path.join(validation_input_dir, "images")
		# validation_input_dir_pnml = os.path.join(validation_input_dir, "pnml")

		# validation_original_dir = os.path.join(validation_saved_dir, "original_net")
		# validation_original_dir_imgs = os.path.join(validation_original_dir, "images")
		# validation_original_dir_pnml = os.path.join(train_original_dir, "pnml")

		# # validation_reconstructed_dir = os.path.join(validation_saved_dir, "reconstructed_net")
		# # validation_reconstructed_dir_imgs = os.path.join(validation_reconstructed_dir, "images")
		# # validation_reconstructed_dir_pnml = os.path.join(validation_reconstructed_dir, "pnml")

		# validation_dirs =[
		# 	validation_graphs_dir, validation_logs_dir, validation_alpha_relations_dir,
		# 	validation_nodes_dir, validation_raw_dir, validation_variants_dir,
		# 	validation_saved_dir, validation_input_dir, validation_input_dir_imgs, validation_input_dir_pnml,
		# 	validation_original_dir, validation_original_dir_imgs, validation_original_dir_pnml]
		# 	# validation_reconstructed_dir, validation_reconstructed_dir_imgs, validation_reconstructed_dir_pnml]


		# test dirs
		test_graphs_dir = os.path.join(self.data_dir, "test_graphs")
		test_logs_dir = os.path.join(test_graphs_dir, "logs")
		test_nodes_dir = os.path.join(test_graphs_dir, "nodes")
		test_raw_dir = os.path.join(test_graphs_dir, "raw")
		test_variants_dir = os.path.join(test_graphs_dir, "variants")
		test_alpha_relations_dir = os.path.join(test_graphs_dir, "alpha_relations")
		test_order_dir = os.path.join(test_graphs_dir, "order")
		test_next_dir = os.path.join(test_graphs_dir, "next")
		test_prev_dir = os.path.join(test_graphs_dir, "prev")

		test_saved_dir = os.path.join(test_graphs_dir, "saved_images_pnml")
		test_input_dir = os.path.join(test_saved_dir, "input_nets")
		test_input_dir_imgs = os.path.join(test_input_dir, "images")
		test_input_dir_pnml = os.path.join(test_input_dir, "pnml")

		test_original_dir = os.path.join(test_saved_dir, "original_nets")
		test_original_dir_imgs = os.path.join(test_original_dir, "images")
		test_original_dir_pnml = os.path.join(test_original_dir, "pnml")

		# test_reconstructed_dir = os.path.join(test_saved_dir, "reconstructed_nets")
		# test_reconstructed_dir_imgs = os.path.join(test_reconstructed_dir, "images")
		# test_reconstructed_dir_pnml = os.path.join(test_reconstructed_dir, "pnml")

		test_dirs = [test_graphs_dir, test_logs_dir, test_nodes_dir, test_raw_dir, test_variants_dir,
			test_saved_dir, test_input_dir, test_input_dir_imgs, test_input_dir_pnml,
			test_alpha_relations_dir, test_original_dir, test_original_dir_imgs, test_original_dir_pnml,
                    test_order_dir, test_next_dir, test_prev_dir]
			# test_reconstructed_dir, test_reconstructed_dir_imgs, test_reconstructed_dir_pnml]

		

		# all_dirs = train_dirs + validation_dirs + test_dirs
		all_dirs = train_dirs + test_dirs

		create_dirs(all_dirs)

		# actual split
		train_graphs  = train_df['log'].unique()
		# valid_graphs = valid_df['log'].unique()
		test_graphs = test_df['log'].unique()

		# split and move nodes ######################################################################################
		split_move_files(
			os.path.join(self.graphs_dir, "nodes"),
			# [train_nodes_dir,validation_nodes_dir,test_nodes_dir], 
			[train_nodes_dir, test_nodes_dir],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)


		# split and move variants ##################################################################################
		split_move_files(
			os.path.join(self.graphs_dir, "variants"),
			# [train_variants_dir,validation_variants_dir,test_variants_dir], 
			[train_variants_dir, test_variants_dir],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)


		# split and move logs #####################################################################################
		split_move_files(
			os.path.join(self.graphs_dir, "logs"),
			# [train_logs_dir,validation_logs_dir,test_logs_dir], 
			[train_logs_dir, test_logs_dir],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)


		# split and move raw ######################################################################################
		edge_indices = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'gaph' not in file]

		split_move_files(
			os.path.join(self.graphs_dir, "raw"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_raw_dir, test_raw_dir],
			train_graphs, test_graphs,
			# train_graphs, valid_graphs, test_graphs,
			edge_indices)


		originals = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'original' in file]

		split_move_files(
			os.path.join(self.graphs_dir, "raw"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_raw_dir, test_raw_dir],
			train_graphs, test_graphs,
			# train_graphs, valid_graphs, test_graphs,
			originals)


		xs = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'x' in file]

		split_move_files(
			os.path.join(self.graphs_dir, "raw"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_raw_dir, test_raw_dir],
			train_graphs, test_graphs,
			# train_graphs, valid_graphs, test_graphs,
			xs)


		# ys = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'y' in file]

		# split_move_files(
		# 	os.path.join(self.graphs_dir, "raw"),
		# 	[train_raw_dir,validation_raw_dir,test_raw_dir], 
		# 	train_graphs, valid_graphs, test_graphs,
		# 	ys)

		# split and move order
		split_move_files(
			os.path.join(self.graphs_dir, "order"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_order_dir, test_order_dir],
			train_graphs, test_graphs)

		# split and move next
		split_move_files(
			os.path.join(self.graphs_dir, "next"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_next_dir, test_next_dir],
			train_graphs, test_graphs)
		
		# split and move prev
		split_move_files(
			os.path.join(self.graphs_dir, "prev"),
			# [train_raw_dir,validation_raw_dir,test_raw_dir], 
			[train_prev_dir, test_prev_dir],
			train_graphs, test_graphs)

		# split and move imgs and pnml ######################################################################################
		split_move_files(
			self.input_dir_imgs,
			# [train_input_dir_imgs, validation_input_dir_imgs, test_input_dir_imgs], 
			[train_input_dir_imgs, test_input_dir_imgs],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)
		
		split_move_files(
			self.input_dir_pnml,
			# [train_input_dir_pnml, validation_input_dir_pnml, test_input_dir_pnml], 
			[train_input_dir_pnml, test_input_dir_pnml],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)

		split_move_files(
			self.original_dir_imgs,
			# [train_original_dir_imgs, validation_original_dir_imgs, test_original_dir_imgs], 
			[train_original_dir_imgs, test_original_dir_imgs],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)
			
		split_move_files(
			self.original_dir_pnml,
			# [train_original_dir_pnml, validation_original_dir_pnml, test_original_dir_pnml], 
			[train_original_dir_pnml, test_original_dir_pnml],
			train_graphs, test_graphs
			# train_graphs, valid_graphs, test_graphs
			)
	
		# split_move_files(
		#     self.reconstr_dir_imgs,
		#     [train_reconstructed_dir_imgs, validation_reconstructed_dir_imgs, test_reconstructed_dir_imgs], 
		#     train_graphs, valid_graphs, test_graphs
		#     )
		#     
		# split_move_files(
		#     self.reconstr_dir_pnml,
		#     [train_reconstructed_dir_pnml, validation_reconstructed_dir_pnml, test_reconstructed_dir_pnml], 
		#     train_graphs, valid_graphs, test_graphs
		#     )

		# split and move alpha relations ####################################################################################
		split_move_files(
			os.path.join(self.graphs_dir, "alpha_relations"),
			# [train_alpha_relations_dir,validation_alpha_relations_dir,test_alpha_relations_dir], 
			[train_alpha_relations_dir, test_alpha_relations_dir],
			# train_graphs, valid_graphs, test_graphs
			train_graphs, test_graphs
			)

		
		# assertions
		no_train = len(train_graphs)
		no_test = len(test_graphs)
		assert len(os.listdir(train_logs_dir)) == no_train
		assert len(os.listdir(train_nodes_dir)) == no_train
		assert len(os.listdir(train_raw_dir)) == no_train * 3
		assert len(os.listdir(train_variants_dir)) == no_train
		assert len(os.listdir(train_alpha_relations_dir)) == no_train
		assert len(os.listdir(train_input_dir_imgs)) == no_train
		assert len(os.listdir(train_input_dir_pnml)) == no_train
		assert len(os.listdir(train_original_dir_imgs)) == no_train
		assert len(os.listdir(train_original_dir_pnml)) == no_train
		assert len(os.listdir(train_order_dir)) == no_train
		assert len(os.listdir(train_next_dir)) == no_train
		assert len(os.listdir(test_logs_dir)) == no_test
		assert len(os.listdir(test_nodes_dir)) == no_test
		assert len(os.listdir(test_raw_dir)) == no_test * 3
		assert len(os.listdir(test_variants_dir)) == no_test
		assert len(os.listdir(test_alpha_relations_dir)) == no_test
		assert len(os.listdir(test_input_dir_imgs)) == no_test
		assert len(os.listdir(test_input_dir_pnml)) == no_test
		assert len(os.listdir(test_original_dir_imgs)) == no_test
		assert len(os.listdir(test_original_dir_pnml)) == no_test
		assert len(os.listdir(test_order_dir)) == no_test
		assert len(os.listdir(test_next_dir)) == no_test
		# assertions


		shutil.rmtree(self.graphs_dir)
		shutil.rmtree(self.saved_dir)


	def clean_dataset_and_split(self):
		ill_formed = self.detect_ill_formed()
		
		original_df = pd.read_csv(os.path.join(self.data_dir, 'tmp_data.csv'))
		graph_stats_df, removed = self.build_graph_stats_df(ill_formed)

		print(f'remaining graphs {len(os.listdir(self.logs_dir))-len(removed)}/{len(os.listdir(self.logs_dir))}')

		# train_df, valid_df, test_df = self.sample(original_df, graph_stats_df, removed)
		train_df, test_df = self.sample(original_df, graph_stats_df, removed)

		print(f"train samples: {len(train_df['log'].unique())}")
		# print(f"validation samples: {len(valid_df['log'].unique())}")
		print(f"test samples: {len(test_df['log'].unique())}")

		# print(f'remaining number of rows {len(train_df)+len(valid_df)+len(test_df)}/{len(original_df)}')
		print(f'remaining number of rows {len(train_df)+len(test_df)}/{len(original_df)}')

		# self.split(train_df, valid_df, test_df)
		self.split(train_df, test_df)

		# df = pd.concat([train_df, valid_df, test_df], axis=0)
		df = pd.concat([train_df, test_df], axis=0)
		df = df.loc[:, ~df.columns.isin(['bin', 'mean_deg'])]
		df.reset_index(drop=True, inplace=True)
		df.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)