from pathlib import Path
import sys
import time

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import pandas as pd
import os
import threading
from utils.pm4py_utils import evaluate, load_log_xes, load_petri_net


class Evaluator():
    def __init__(self, base_dir, model_type, timeout=60):
        self.base_dir = base_dir
        self.model_type = model_type
        self.timeout = timeout
        self.fitness = {}
        self.precision = {}
        self.generalization = {}
        self.simplicity = {}
        self.threads = []
        self.df = None

    
    def get_results(self):
        return self.df
    

    def compute_statistics(self, idx, log, net, im, fm):
        fitness, precision, generalization, simplicity = evaluate(log, net, im, fm)
        self.fitness[idx] = fitness
        self.precision[idx] = precision
        self.generalization[idx] = generalization
        self.simplicity[idx] = simplicity

    def evaluate(self, miner=False):
        logs_dir = os.path.join(self.base_dir, "logs")
        nets_dir = os.path.join(self.base_dir, "inference","pnml")
        log_names = sorted(os.listdir(logs_dir))
        net_names = sorted(os.listdir(nets_dir))

        get_idx = lambda name: int(name.split('_')[1].split('.')[0])

        logs = []
        for log_name, net_name in zip(log_names, net_names):
            log = load_log_xes(os.path.join(logs_dir, log_name))
            logs.append(log)

        for i, log_name, net_name in zip(range(len(log_names)), log_names, net_names):
            idx = get_idx(log_name)

            log = logs[i]
            if miner:
                import pm4py
                net, im, fm = pm4py.discover_petri_net_alpha(log)
            else:
                net, im, fm = load_petri_net(os.path.join(nets_dir, net_name))
      
            self.compute_statistics(idx, log, net, im, fm)

        self.build_df()

        for thread in self.threads:
            assert thread.is_alive() == False
        
        miner_name = "_alpha" if miner else ''
        
        mean = self.df[["perc_fit_traces","average_trace_fitness","precision","generalization","simplicity"]].mean(axis=0)
        mean.to_csv(os.path.join(self.base_dir, "inference", f"mean_results_{self.model_type}{miner_name}.txt"), sep='\t')
        self.df.to_csv(os.path.join(self.base_dir, "inference", f"results_{self.model_type}{miner_name}.csv"))


    def build_df(self):
        self.df = pd.DataFrame(
            columns = ['idx', 'perc_fit_traces','average_trace_fitness','precision', 'generalization', 'simplicity'])
        
        assert len(self.fitness) == len(self.precision)
        assert len(self.precision) == len(self.generalization)
        assert len(self.generalization) == len(self.simplicity)
        
        keys = self.fitness.keys()
        for idx in keys:
            row = {
                'idx':idx,
                'perc_fit_traces':self.fitness[idx]['perc_fit_traces'],
                'average_trace_fitness':self.fitness[idx]['average_trace_fitness'],
                'precision':self.precision[idx],
                'generalization':self.generalization[idx],
                'simplicity':self.simplicity[idx],
                }
            self.df = self.df.append(row, ignore_index=True)
        
        self.df.reset_index(drop=True, inplace=True)


        