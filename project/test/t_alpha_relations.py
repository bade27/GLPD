from pathlib import Path
import sys

from tqdm import tqdm


path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import pickle
import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan

def get_alpha_relations(log):
    variants = pm4py.get_variants(log)
    direct_succession = {}
    for variant in variants:
        for i, t in enumerate(variant):
            if t not in direct_succession: direct_succession[t] = set()
            if i+1 < len(variant): direct_succession[t].add(variant[i+1])

    causality_fn = lambda x, y: y in direct_succession[x] and not x in direct_succession[y]
    parallel_fn = lambda x, y: y in direct_succession[x] and x in direct_succession[y]
    choice_fn = lambda x, y: y not in direct_succession[x] and not x in direct_succession[y]

    causality = {}
    parallel = {}
    choice = {}
    for variant in variants:
        for i, t in enumerate(variant):
            if t not in causality: causality[t] = set()
            if t not in parallel: parallel[t] = set()
            if t not in choice: choice[t] = set()

            if i+1 < len(variant):
                if causality_fn(t, variant[i+1]):
                    causality[t].add(variant[i+1])
                if parallel_fn(t, variant[i+1]):
                    parallel[t].add(variant[i+1])
                if choice_fn(t, variant[i+1]):
                    choice[t].add(variant[i+1])

parameters = {
    "mode":8,
    "min":4,
    "max":15,
    "sequence": 0.40,
    "choice": 0.35,
    "parallel": 0.25,
    "or": 0.0,
    "silent": 0.0,
    "loop":0.0,
    "no_models":500
    }


trees = pm4py.generate_process_tree(parameters=parameters)

alpha_sound = 0
for elem in tqdm(enumerate(trees)):
    i, tree = elem
    net_ws, im_ws, fm_ws = pm4py.convert_to_petri_net(tree)

    log = pm4py.play_out(net_ws, im_ws, fm_ws, parameters={"no_traces":20})

    # pm4py.write_xes(log, 'C:\\Users\\matte\\Desktop\\exported.xes')

    # log = pm4py.read_xes('C:\\Users\\matte\\Desktop\\exported.xes')

    # print(log[0][0])


    # log = generate_log(tree, 10)

    # save_log_xes(log, "C:\\Users\\matte\\Desktop\\log.xes")

    net, im, fm = pm4py.discover_petri_net_alpha(log)

    def is_sound(net, im, fm):

        return woflan.apply(net, im, fm, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                         woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                         woflan.Parameters.RETURN_DIAGNOSTICS: False})


    if is_sound(net, im, fm):
        pm4py.write_pnml(net, im, fm, f"C:\\Users\\matte\\Desktop\\sound_dataset\\petri_{i}.pnml")
        pm4py.save_vis_petri_net(net, im, fm, f"C:\\Users\\matte\\Desktop\\sound_dataset\\net_{i}.svg")

        with open(f"C:\\Users\\matte\\Desktop\\sound_dataset\\alpha_{i}.pkl", "wb") as file:
            pickle.dump(get_alpha_relations(log), file)

        log = pm4py.play_out(net, im, fm, parameters={"no_traces":20})
        pm4py.write_xes(log, f'C:\\Users\\matte\\Desktop\\sound_dataset\\exported_{i}.xes')

        alpha_sound += 1

print(f"discovered {alpha_sound} sound nets over {len(trees)}")

# pm4py.view_petri_net(net, im, fm)
# pm4py.view_petri_net(net1, im1, fm1)

    