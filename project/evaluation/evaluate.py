import threading

from project.utils.pm4py_utils import evaluate


class Evaluator():
    def __init__(self, timeout=120):
        self.timeout = timeout
        self.fitness = {}
        self.precision = {}
        self.generalization = {}
        self.simplicity = {}
        self.threads = []
        self.df = None

    def compute_statistics(self, idx, log, net, im, fm, t_stop):
        fitness, precision, generalization, simplicity = evaluate(log, net, im, fm)
        if not t_stop.is_set():
            self.fitness[idx] = fitness
            self.precision[idx] = precision
            self.generalization[idx] = generalization
            self.simplicity[idx] = simplicity

    def evaluate(self, logs, nets):
        for log, net in zip(logs, nets):
            pn , im, fm = net
            idx = 0 # set idx

            t_stop = threading.Event()
            t = threading.Thread(target=self.compute_statistics, args=(idx, log, pn, im, fm, t_stop))
            t.start()
            self.threads.append(t)

            t.join(self.timeout + self.timeout//2)
            if t.is_alive():
                t_stop.set()

            t.join()

        self.build_df()
        
        return self.df

    def buid_df(self):
        pass

        