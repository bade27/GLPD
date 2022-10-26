from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import os
import torch
from data_handling import dataset
import pm4py
from utils.general_utils import load_pickle
from utils.general_utils import create_dirs
from model.self_supervised import SelfSupPredictor
from model import structure as st
from utils.graph_utils import add_silent_transitions
from utils.pm4py_utils import save_petri_net_to_img, save_petri_net_to_pnml
from utils.petri_net_utils import back_to_petri
from pm4py.algo.evaluation import algorithm as evaluator
import json


def infer(data_dir, log_filename, model_filename, silent_transitions=False, preprocessing=False):
	if preprocessing:
		preprocessor = dataset.Dataset(data_dir, type_of_features="temporal", synth=False, filename=log_filename)
		preprocessor.generate_dataset(save_images=True, save_models=True)

	destination_dir = data_dir + "/inference"
	if not os.path.exists(destination_dir):
		os.mkdir(destination_dir)

	raw_dir = os.path.join(data_dir, "graphs", "raw")
	nodes_dir = os.path.join(data_dir, "graphs", "nodes")
	logs_dir = os.path.join(data_dir, "graphs", "logs")
	alpha_relations_dir = os.path.join(data_dir, "graphs", "alpha_relations")
	next_dir = os.path.join(data_dir, "graphs", "next")
	prev_dir = os.path.join(data_dir, "graphs", "prev")
	order_dir = os.path.join(data_dir, "graphs", "order")
	variants_dir = os.path.join(data_dir, "graphs", "variants")

	x_filename = [file for file in os.listdir(raw_dir) if "x" in file][0]
	original_filename = [file for file in os.listdir(raw_dir) if "original" in file][0]
	edge_index_filename = [file for file in os.listdir(raw_dir) if "graph" in file][0]
	nodes_filename = os.listdir(nodes_dir)[0]
	logs_filename = os.listdir(logs_dir)[0]
	next_filename = os.listdir(next_dir)[0]
	prev_filename = os.listdir(prev_dir)[0]
	order_filename = os.listdir(order_dir)[0]
	variants_filename = os.listdir(variants_dir)[0]
	alpha_relations_filename = os.listdir(alpha_relations_dir)[0]

	x = torch.load(os.path.join(raw_dir, x_filename))
	original = torch.load(os.path.join(raw_dir, original_filename))
	edge_index = torch.load(os.path.join(raw_dir, edge_index_filename))
	nodes = load_pickle(os.path.join(nodes_dir, nodes_filename))
	log = pm4py.read_xes(os.path.join(logs_dir, logs_filename))
	nextt = load_pickle(os.path.join(next_dir, next_filename))
	prev = load_pickle(os.path.join(prev_dir, prev_filename))
	order = load_pickle(os.path.join(order_dir, order_filename))
	variants = load_pickle(os.path.join(variants_dir, variants_filename))
	alpha_relations = load_pickle(os.path.join(alpha_relations_dir, alpha_relations_filename))

	img_dir = os.path.join(destination_dir, "images")
	pnml_dir = os.path.join(destination_dir, "pnml")

	create_dirs([img_dir, pnml_dir])

	num_node_features, features_size = st.num_node_features, st.features_size
	output_size = st.output_size_self_sup
	model = SelfSupPredictor(num_node_features, features_size, output_size, "gcn", "cpu")
	model.load_state_dict(torch.load(model_filename))
	model.eval()

	print("discovering model...", end=' ')
	places = model(x, edge_index, original, nodes, variants, order, nextt)

	mask = ['place' not in n for n in nodes]
	for place in places:
		mask[place] = True

	assert sum(mask[:nodes.index('|')+1]) == nodes.index('|')+1

	print(f"discovered {sum(mask[nodes.index('|')+1:])}/{len(mask[nodes.index('|')+1:])} places")

	net, im, fm = back_to_petri(original, nodes, mask)

	save_petri_net_to_img(net, im, fm, os.path.join(img_dir, 'net.png'))
	save_petri_net_to_pnml(net, im, fm, os.path.join(pnml_dir, 'net.pnml'))

	evaluation = evaluator.apply(log, net, im, fm)
	with open(os.path.join(destination_dir, "evaluation.txt"), "w") as file:
		file.write(json.dumps(evaluation))

	with open(os.path.join(destination_dir, "info.txt"), "w") as file:
		no_places = sum(mask[nodes.index('|')+1:])/len(mask[nodes.index('|')+1:])
		no_silent = len([node for node in nodes if "silent" in nodes])
		file.write(json.dumps({"no_places":no_places,"no_silent":no_silent}))

	if silent_transitions:
		print("adding silent transitions...", end=' ')
		s_img_dir = os.path.join(destination_dir+"_silent", "images")
		s_pnml_dir = os.path.join(destination_dir+"_silent", "pnml")

		create_dirs([s_img_dir, s_pnml_dir])

		new_edge_index, new_nodes, new_mask = add_silent_transitions(original, nextt, prev, mask, nodes, alpha_relations)

		silent = [st for st in nodes if "silent" in st]
		print(f"{len(silent)} silent transitions added")

		s_net, s_im, s_fm = back_to_petri(new_edge_index, new_nodes, new_mask)

		save_petri_net_to_img(s_net, s_im, s_fm, os.path.join(s_img_dir, 'net.png'))
		save_petri_net_to_pnml(s_net, s_im, s_fm, os.path.join(s_pnml_dir, 'net.pnml'))

		s_evaluation = evaluator.apply(log, s_net, s_im, s_fm)
		with open(os.path.join(destination_dir+"_silent", "silent_evaluation.txt"), "w") as file:
			file.write(json.dumps(s_evaluation))

		with open(os.path.join(destination_dir+"_silent", "info.txt"), "w") as file:
			no_places = sum(new_mask[new_nodes.index('|')+1:])/len(new_mask[new_nodes.index('|')+1:])
			no_silent = len([node for node in new_nodes if "silent" in new_nodes])
			file.write(json.dumps({"no_places":no_places,"no_silent":no_silent}))



# data_dir = "/home/linuxpc/MEGAsync/classical_miners/BPI_2012/"
# log_filename = "BPI_Challenge_2012_reducted.xes"
# model_filename = "/home/linuxpc/Documenti/TESI/best_model/self_supervised_97.pt"
# infer(data_dir, log_filename, model_filename, silent_transitions=True)

real_logs_dir = "D:\\Vario\\TESI_IMMAGINI_E_RISULTATI\\my_project\\"
model_path = "D:\\Vario\\TESI_IMMAGINI_E_RISULTATI\\trained_model\\best_model\\self_supervised_97.pt"
for folder in os.listdir(real_logs_dir):
	log_filename = [file for file in os.listdir(os.path.join(real_logs_dir, folder)) if "xes" in file][0]
	dirdir = os.path.join(real_logs_dir, folder)
	infer(dirdir, log_filename, model_path, silent_transitions=True, preprocessing=True)