from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import pm4py
from pm4py.algo.simulation.tree_generator import algorithm as tree_gen
from pm4py.objects.process_tree import semantics
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.petri_net import visualizer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import write_pnml, view_petri_net, save_vis_petri_net
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation import replay_fitness
from pm4py.algo.discovery.alpha import algorithm as alpha
from pm4py.algo.analysis.woflan import algorithm as woflan


def generate_trees(parameters):
    trees = tree_gen.apply(parameters=parameters)
    return trees


def generate_log(tree, no_traces):
    log = semantics.generate_log(tree, no_traces=no_traces)
    return log


def log_to_dataframe(log):
    dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    if "time:timestamp" not in dataframe:
        dataframe["time:timestamp"] = range(len(dataframe))
    dataframe = dataframe.drop(dataframe.columns[~dataframe.columns.isin(["concept:name", "time:timestamp", "case:concept:name"])],axis=1)
    dataframe = dataframe.rename({"concept:name":"activity","time:timestamp":"timestamp","case:concept:name":"case"}, axis=1)
    dataframe["activity"] = dataframe["activity"].apply(lambda x: x.replace(" ", "_"))
    return dataframe


def from_tree_to_petri_net(tree):
    net, im, fm = pt_converter.apply(tree)
    return net, im, fm


def build_petri_net_from_log(log, miner=alpha):
    net, im, fm = miner.apply(log)
    return net, im, fm


def display_petri_net(net, initial_marking, final_marking, format=None):
    if format:
        view_petri_net(net, initial_marking, final_marking, format=format)
    else:
        view_petri_net(net, initial_marking, final_marking)


def save_petri_net_to_pnml(net, initial_marking, final_marking, path):
    write_pnml(net, initial_marking, final_marking, path)


def save_petri_net_to_img(net, initial_marking, final_marking, path):
    save_vis_petri_net(net, initial_marking, final_marking, path)


def save_log_xes(log, path):
    pm4py.write_xes(log, path)


def load_log_xes(path):
    log = pm4py.read_xes(path)
    return log


def load_petri_net(path):
    net, initial_marking, final_marking = pm4py.read_pnml(path)
    return net, initial_marking, final_marking


def evaluate(log, net, im, fm):
  fitness = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
  precision = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
  generalization = generalization_evaluator.apply(log, net, im, fm)
  simplicity = simplicity_evaluator.apply(net)

  return fitness, precision, generalization, simplicity


def is_sound(net, im, fm):
  is_sound = woflan.apply(net, im, fm, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                  woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                  woflan.Parameters.RETURN_DIAGNOSTICS: False})
  return is_sound


def get_variants(log, reverse=True):
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=reverse)
    return variants_count


def get_variants_parsed(log, top=None):
    if top != None:
        filtered_log = pm4py.filter_variants_top_k(log, top)
        log = filtered_log
        
    variants_log = pm4py.get_variants_as_tuples(log)
    variants = list()
    for variant_log in variants_log:
        vl = [activity.replace(" ", "_") for activity in list(variant_log)]
        vl = ['>'] + vl + ['|']
        variants.append(vl)

    return variants


def get_variants_distribution_parsed(log):
    variants_count = case_statistics.get_variant_statistics(log)
    variants_count = sorted(variants_count, key=lambda x: x['count'], reverse=True)

    total = sum([variant["count"] for variant in variants_count])
    variants = []
    for variant in variants_count:
        v = variant['variant'].split(',')
        v.insert(0, '>')
        v.append('|')
        c = variant['count']
        d = {'variant': v, 'count': c/total}
        variants.append(d)

    return variants



def filter_log_perc(log, min_perc=0.2):
    filtered_log = pm4py.filter_variants_by_coverage_percentage(log, min_perc)
    return filtered_log


def filter_log_top_k(log, k=30):
    filtered_log = pm4py.filter_variants_top_k(log, k)
    return filtered_log