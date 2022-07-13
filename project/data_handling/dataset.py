from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)



import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import networkx as nx
from utils.general_utils import create_dirs, dump_to_pickle, create_dirs, split_move_files
from utils.graph_utils import build_arcs_dataframe, build_graph, build_temp_graph, generate_features, draw_networkx, get_backward_star, get_forward_star, mean_node_degree
from utils.petri_net_utils import add_places, alpha_relations, build_net_from_places, find_actual_places
from utils.pm4py_utils import display_petri_net, generate_trees, generate_log, build_petri_net_from_log, get_variants_parsed, log_to_dataframe, save_log_xes, save_petri_net_to_img, save_petri_net_to_pnml
import model.structure as model_structure
import shutil
from sklearn.preprocessing import KBinsDiscretizer
import torch


class Dataset():
    def __init__(self, data_dir, random_features=False):
        self.data_dir = data_dir
        self.random_features = random_features

        self.graphs_dir = os.path.join(self.data_dir, "graphs")
        self.logs_dir = os.path.join(self.graphs_dir, "logs")
        self.raw_dir = os.path.join(self.graphs_dir, "raw")
        self.graph_nodes_dir = os.path.join(self.graphs_dir, "nodes")
        self.graph_variants_dir = os.path.join(self.graphs_dir, "variants")
        self.saved_dir = os.path.join(self.data_dir, "saved_images_pnml")
        self.networkx_dir = os.path.join(self.data_dir, 'networkx')

        self.original_dir = os.path.join(self.saved_dir, "original_nets")
        self.original_dir_imgs = os.path.join(self.original_dir, "images")
        self.original_dir_pnml = os.path.join(self.original_dir, "pnml")

        self.input_dir = os.path.join(self.saved_dir, "input_net")
        self.input_dir_imgs = os.path.join(self.input_dir, "images")
        self.input_dir_pnml = os.path.join(self.input_dir, "pnml")

        # self.reconstr_dir = os.path.join(self.saved_dir, "reconstructed_nets")
        # self.reconstr_dir_imgs = os.path.join(self.reconstr_dir, "images")
        # self.reconstr_dir_pnml = os.path.join(self.reconstr_dir, "pnml")

        self.dirs = [
            self.data_dir, self.graphs_dir, 
            self.logs_dir, self.raw_dir, self.graph_nodes_dir, self.graph_variants_dir,
            self.networkx_dir, self.saved_dir, 
            self.original_dir, self.original_dir_imgs, self.original_dir_pnml,
            self.input_dir, self.input_dir_imgs, self.input_dir_pnml] 
            # self.reconstr_dir, self.reconstr_dir_imgs, self.reconstr_dir_pnml]

        create_dirs(self.dirs)

        self.statistics = None
        self.set_statistics()

        self.columns = ["source", "destination", "timestamp", "case", "log", "state_label"]

    def get_encoding(self):
        encoding = {}
        encoding["<"] = 0
        for k in range(97,123):
            encoding[chr(k)] = k-97+1
        encoding["|"] = 27

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


    def generate_dataset(self, save_networkx = False, save_images = False, save_models = False, visualize_nets = False):

        encoding = self.get_encoding()

        dataset_l = []

        statistics = self.statistics

        pad = len(str(statistics["no_models"]*statistics['no_datasets']))
        offset = 0

        total_csv_entries = 0
    
        for c in range(statistics['no_datasets']):

            parameters = {k:v[c] for k, v in statistics.items() if k not in ['no_models', 'no_traces', 'no_datasets', 'no_features']}
            parameters.update({'no_models': statistics['no_models']})

            no_traces = statistics['no_traces']

            trees = generate_trees(parameters)

            data = pd.DataFrame()
    
            for i in tqdm(range(parameters["no_models"])):
                # log and initial df
                tree = trees[i] if parameters["no_models"] > 1 else trees
                log = generate_log(tree, no_traces)
            
                df = log_to_dataframe(log)
                net, im, fm = build_petri_net_from_log(log)
        
                unique_activities = set([transition.name for transition in net.transitions])

                # insert special activities '<' (start) and '|' (finish)
                df_start_finish = pd.DataFrame()
                for case in df["case"].unique():
                    current_case = df[df["case"]==case].copy()
                    df_start = pd.DataFrame({"activity":'<', "timestamp":None, "case":case},index=[0])
                    df_finish = pd.DataFrame({"activity":'|', "timestamp":None, "case":case},index=[0])
                    df_start_finish = pd.concat([df_start_finish, df_start], axis=0)
                    df_start_finish = pd.concat([df_start_finish, current_case], axis=0)
                    df_start_finish = pd.concat([df_start_finish, df_finish], axis=0)
                    df_start_finish.reset_index(drop=True,inplace=True)

                unique_activities.add('<')
                unique_activities.add('|')

                # new dataframe to manipulate
                new_df = build_arcs_dataframe(df_start_finish)

                # alpha relations
                df_alpha_relations = alpha_relations(log)

                net_places = find_actual_places(net) # places of the original net
                places = add_places(new_df, df_alpha_relations) # places of the input net

                # input net
                input_net, input_im, input_fm = build_net_from_places(unique_activities, places)

                # networkx graph of the new net
                graph, y, nodes, _ = build_graph(unique_activities, places, net_places, encoding)

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

                assert len(y) >= torch.max(edge_index[0])
                assert len(y) >= torch.max(edge_index[1])

                target = torch.tensor(y)

                assert len(edge_index[0]) == len(edge_index[1])
                assert len(nodes) == len(target)

                # temporal graph
                temporal_graph = build_temp_graph(new_df)
                new_df["log"] = i+offset
                new_df["state_label"] = [0 for _ in range(len(new_df))]
                df_correct_order = new_df[self.columns]
                new_df_features = generate_features(df_correct_order, temporal_graph, statistics['no_features'])
                data = pd.concat([data, new_df_features], axis=0)
                data.reset_index(drop=True, inplace=True)

                # rec_net, rec_im, rec_fm = back_to_petri(original_edge_index, nodes, y)
                
                if visualize_nets:
                    display_petri_net(net, im, fm)
                    display_petri_net(input_net, input_im, input_fm)
                    # display_petri_net(rec_net, rec_im, rec_fm)
                    draw_networkx(graph)

                if save_images:
                    save_petri_net_to_img(net, im, fm, os.path.join(self.original_dir, "images", "original_" + str(i+offset).zfill(pad) + '.png'))
                    save_petri_net_to_img(input_net, input_im, input_fm, os.path.join(self.input_dir, "images", "input_" + str(i+offset).zfill(pad) + '.png'))
                    # save_petri_net_to_img(rec_net, rec_im, rec_fm, os.path.join(self.reconstr_dir, "images", "rec_" + str(i+offset).zfill(pad) + '.png'))

                if save_models:
                    save_petri_net_to_pnml(net, im, fm, os.path.join(self.original_dir, "pnml", "original_" + str(i+offset).zfill(pad) + '.pnml'))
                    save_petri_net_to_pnml(input_net, input_im, input_fm, os.path.join(self.input_dir, "pnml", "input_" + str(i+offset).zfill(pad) + '.pnml'))
                    # save_petri_net_to_pnml(rec_net, rec_im, rec_fm, os.path.join(self.reconstr_dir, "pnml", "rec_" + str(i+offset).zfill(pad) + '.pnml'))

                if save_networkx:
                    dump_to_pickle(os.path.join(self.networkx_dir, 'gx_' + str(i+offset).zfill(pad)), graph)


                variants = get_variants_parsed(log)

                # save files
                torch.save(edge_index, os.path.join(self.raw_dir, "graph_" + str(i+offset).zfill(pad) + ".pt"))
                torch.save(original_edge_index, os.path.join(self.raw_dir, "original_" + str(i+offset).zfill(pad) + ".pt"))
                torch.save(target, os.path.join(self.raw_dir, "y_" + str(i+offset).zfill(pad) + ".pt"))

                if self.random_features:
                    h = torch.max(edge_index)+1
                    w = model_structure.num_node_features
                    no_activities = nodes.index('|')+1
                    no_places = h.item() - no_activities

                    x_activities = torch.randn((no_activities, w))
                    x_activities = torch.nn.functional.normalize(x_activities, p=2.0, dim=1)

                    x_places = torch.zeros((no_places, w))

                    x = torch.cat((x_activities, x_places), dim=0)

                    assert x.shape[0] == h
                    assert x.shape[1] == w

                    torch.save(x, os.path.join(self.raw_dir, "x_" + str(i+offset).zfill(pad) + ".pt"))
                else:
                    # assemble tgn data and move it into self.raw_dir with name x
                    pass



                nodes_file_name = os.path.join(self.graph_nodes_dir, "nodes_" + str(i+offset).zfill(pad))
                dump_to_pickle(nodes_file_name, nodes)

                variants_file_name = os.path.join(self.graph_variants_dir, "variants_" + str(i+offset).zfill(pad))
                dump_to_pickle(variants_file_name, variants)

                save_log_xes(log, os.path.join(self.logs_dir, "log_" + str(i+offset).zfill(pad) + '.xes'))

            # rename activities
            data["source"] = data["source"].map(encoding)
            data["destination"] = data["destination"].map(encoding)

            data["destination"] = data["destination"].astype("int")
            data["timestamp"] = [i for i in range(len(data))]

            dataset_l.append(data)
            total_csv_entries += len(data.index)

            offset += statistics["no_models"]

        total_csv_entries_ctrl = 0
        for d in dataset_l:
            total_csv_entries_ctrl += len(d.index)

        assert total_csv_entries == total_csv_entries_ctrl

        dataset = pd.DataFrame()
        for data in dataset_l:
            dataset = pd.concat([dataset, data], axis=0)
            dataset.reset_index(drop=True, inplace=True)

        assert len(dataset.index) == total_csv_entries

        dataset.to_csv(os.path.join(self.data_dir, 'tmp_data.csv'), index=False)

        assert len(os.listdir(self.logs_dir)) * 4 == len(os.listdir(self.raw_dir))


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
        degrees = []
        removed = []
        idx = 0
        for file in sorted(os.listdir(self.raw_dir)):
            if 'graph_' in file:
                if not ill_formed[idx]:
                    edge_index = torch.load(os.path.join(self.raw_dir, file))
                    nodes = [i for i in range(torch.max(edge_index)+1)]
                    mean_degree = mean_node_degree(nodes, edge_index)
                    if mean_degree is not None:
                        logs.append(idx)
                        degrees.append(mean_degree)
                    else:
                        removed.append(idx)
                else:
                    removed.append(idx)
                idx += 1

        df = pd.DataFrame({'log':logs, 'mean_deg':degrees})
        return df, removed


    def sample(self, dataset, graphs_stat, removed, perc=0.8):
        seed = 1234

        graphs_stat = graphs_stat.drop(removed)

        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        graphs_stat['bin'] = discretizer.fit_transform(graphs_stat[['mean_deg']])

        graphs_stat = graphs_stat.sample(frac=1, random_state=seed)
        graphs_stat.reset_index(drop=True, inplace=True)

        train_dataset = graphs_stat.groupby('bin', group_keys=False).apply(lambda x: x.sample(frac=perc))

        df_all = graphs_stat.merge(train_dataset.drop_duplicates(), on='log', how='left', indicator=True)

        condition = df_all['_merge'] == 'left_only'
        valid_dataset = graphs_stat[condition].groupby('bin', group_keys=False).apply(lambda x: x.sample(frac=0.6))

        test_dataset = graphs_stat.loc[~graphs_stat['log'].isin((set(train_dataset['log'].unique())).union(set(valid_dataset['log'].unique())))]

        assert len(graphs_stat) == len(train_dataset) + len(valid_dataset) + len(test_dataset)

        offset = 0

        train_df = dataset.merge(train_dataset, on='log', how='left')
        train_df = train_df[~train_df['mean_deg'].isna()]
        train_df.reset_index(drop=True, inplace=True)
        train_df['timestamp'] = [i+offset for i in range(len(train_df.index))]

        offset += len(train_df.index)

        valid_df = dataset.merge(valid_dataset, on='log', how='left')
        valid_df = valid_df[~valid_df['mean_deg'].isna()]
        valid_df.reset_index(drop=True, inplace=True)
        valid_df['timestamp'] = [i+offset for i in range(len(valid_df.index))]

        offset += len(valid_df.index)

        test_df = dataset.merge(test_dataset, on='log', how='left')
        test_df = test_df[~test_df['mean_deg'].isna()]
        test_df.reset_index(drop=True, inplace=True)
        test_df['timestamp'] = [i+offset for i in range(len(test_df.index))]

        return train_df, valid_df, test_df


    def split(self, train_df, valid_df, test_df):
        # train dirs
        train_graphs_dir = os.path.join(self.data_dir, "train_graphs")
        train_logs_dir = os.path.join(train_graphs_dir, "logs")
        train_nodes_dir = os.path.join(train_graphs_dir, "nodes")
        train_raw_dir = os.path.join(train_graphs_dir, "raw")
        train_variants_dir = os.path.join(train_graphs_dir, "variants")
        train_dirs = [train_graphs_dir, train_logs_dir, train_nodes_dir, train_raw_dir, train_variants_dir]

        # val dirs
        validation_graphs_dir = os.path.join(self.data_dir, "validation_graphs")
        validation_logs_dir = os.path.join(validation_graphs_dir, "logs")
        validation_nodes_dir = os.path.join(validation_graphs_dir, "nodes")
        validation_raw_dir = os.path.join(validation_graphs_dir, "raw")
        validation_variants_dir = os.path.join(validation_graphs_dir, "variants")
        validation_dirs =[
            validation_graphs_dir, validation_logs_dir, 
            validation_nodes_dir, validation_raw_dir, validation_variants_dir
            ]

        # test dirs
        test_graphs_dir = os.path.join(self.data_dir, "test_graphs")
        test_logs_dir = os.path.join(test_graphs_dir, "logs")
        test_nodes_dir = os.path.join(test_graphs_dir, "nodes")
        test_raw_dir = os.path.join(test_graphs_dir, "raw")
        test_variants_dir = os.path.join(test_graphs_dir, "variants")
        test_dirs = [test_graphs_dir, test_logs_dir, test_nodes_dir, test_raw_dir, test_variants_dir]

        all_dirs = train_dirs + validation_dirs + test_dirs

        create_dirs(all_dirs)

        # actual split
        train_graphs  = train_df['log'].unique()
        valid_graphs = valid_df['log'].unique()
        test_graphs = test_df['log'].unique()

        # split and move nodes ######################################################################################
        split_move_files(
            os.path.join(self.graphs_dir, "nodes"),
            [train_nodes_dir,validation_nodes_dir,test_nodes_dir], 
            train_graphs, valid_graphs, test_graphs
            )


        # split and move variants ##################################################################################
        split_move_files(
            os.path.join(self.graphs_dir, "variants"),
            [train_variants_dir,validation_variants_dir,test_variants_dir], 
            train_graphs, valid_graphs, test_graphs
            )


        # split and move logs #####################################################################################
        split_move_files(
            os.path.join(self.graphs_dir, "logs"),
            [train_logs_dir,validation_logs_dir,test_logs_dir], 
            train_graphs, valid_graphs, test_graphs
            )


        # split and move raw ######################################################################################
        edge_indices = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'gaph' not in file]

        split_move_files(
            os.path.join(self.graphs_dir, "raw"),
            [train_raw_dir,validation_raw_dir,test_raw_dir], 
            train_graphs, valid_graphs, test_graphs,
            edge_indices)


        originals = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'original' in file]

        split_move_files(
            os.path.join(self.graphs_dir, "raw"),
            [train_raw_dir,validation_raw_dir,test_raw_dir], 
            train_graphs, valid_graphs, test_graphs,
            originals)


        xs = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'x' in file]

        split_move_files(
            os.path.join(self.graphs_dir, "raw"),
            [train_raw_dir,validation_raw_dir,test_raw_dir], 
            train_graphs, valid_graphs, test_graphs,
            xs)


        ys = [file for file in sorted(os.listdir(os.path.join(self.graphs_dir, 'raw'))) if 'y' in file]

        split_move_files(
            os.path.join(self.graphs_dir, "raw"),
            [train_raw_dir,validation_raw_dir,test_raw_dir], 
            train_graphs, valid_graphs, test_graphs,
            ys)

        shutil.rmtree(self.graphs_dir)


    def clean_dataset_and_split(self):
        ill_formed = self.detect_ill_formed()
        
        original_df = pd.read_csv(os.path.join(self.data_dir, 'tmp_data.csv'))
        graph_stats_df, removed = self.build_graph_stats_df(ill_formed)

        print(f'remaining graphs {len(os.listdir(self.logs_dir))-len(removed)}/{len(os.listdir(self.logs_dir))}')

        train_df, valid_df, test_df = self.sample(original_df, graph_stats_df, removed)

        print(f"train samples: {len(train_df['log'].unique())}")
        print(f"validation samples: {len(valid_df['log'].unique())}")
        print(f"test samples: {len(test_df['log'].unique())}")

        print(f'remaining number of rows {len(train_df)+len(valid_df)+len(test_df)}/{len(original_df)}')

        self.split(train_df, valid_df, test_df)

        df = pd.concat([train_df, valid_df, test_df], axis=0)
        df = df.loc[:, ~df.columns.isin(['bin', 'mean_deg'])]
        df.reset_index(drop=True, inplace=True)
        df.to_csv(os.path.join(self.data_dir, 'data.csv'), index=False)