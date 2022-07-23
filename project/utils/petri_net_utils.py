from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


import re
import itertools
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from utils.pm4py_utils import get_variants_parsed


def get_eventually_follows(log, depth=1):
    variants = get_variants_parsed(log)
    eventually_follows = {}
    for d in range(0, depth):
        for variant in variants:
            for i, t in enumerate(variant):
                if t not in eventually_follows: eventually_follows[t] = set()
                if i+d+1 < len(variant): 
                    eventually_follows[t].add(variant[i+d+1])
    
    return eventually_follows


def get_alpha_relations(log, depth=2):
    variants = get_variants_parsed(log)
    
    direct_succession = get_eventually_follows(log)
    eventually_follows = get_eventually_follows(log, depth)

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
    dict_alpha_relations = {
        "direct_succession": direct_succession,
        "eventually_follows": eventually_follows,
        "causality":causality,
        "parallel":parallel,
        "choice":choice}
    return dict_alpha_relations


def add_one_to_one(follow_relation, places):
    new_places = set()
    new_places = new_places.union(places)
    for src, dsts in follow_relation.items():
        for dst in sorted(list(dsts)):
            s = src + " ---> "
            s += dst + " "
            new_places.add(s[:-1])
            s = ""

    return new_places


def add_one_to_many(follow_relation, parallel, places):
    new_places = set()
    new_places = new_places.union(places)
    
    for src, dsts in follow_relation.items():
        possible_outgoing_activities = []

        parallel_successors = set()
        for dst in dsts:
            if src in parallel and dst in parallel[src]:
                parallel_successors.add(dst)

        outgoing = dsts.difference(parallel_successors)
        
        for n in range(2, len(outgoing)+1):
            comb = itertools.combinations(outgoing, n)
            for c in comb:
                possible_outgoing_activities.append(c)
        cross_prod = itertools.product([src], possible_outgoing_activities)

        for cp in cross_prod:
            s = cp[0] + ' ---> '
            for x in cp[1]:
                s += x + ' '
            new_places.add(s[:-1])
    return new_places


def add_many_to_one(follow_relation, parallel, places):
    new_places = set()
    new_places = new_places.union(places)

    direct_predecessors = {}
    for activity in follow_relation.keys():
        if activity not in direct_predecessors: direct_predecessors[activity] = set()
        for src, dsts in follow_relation.items():
            if activity in dsts:
                direct_predecessors[activity].add(src)
    
    for dst, srcs in direct_predecessors.items():
        possible_incoming_activities = []

        parallel_predecessors = set()
        for src in srcs:
            if dst in parallel and src in parallel[dst]:
                parallel_predecessors.add(src)

        incoming = dsts.difference(parallel_predecessors)
        
        for n in range(2, len(incoming)+1):
            comb = itertools.combinations(incoming, n)
            for c in comb:
                possible_incoming_activities.append(c)
        cross_prod = itertools.product(possible_incoming_activities, [dst])

        for cp in cross_prod:
            s = '---> ' + cp[1]
            for x in cp[0]:
                s = x + ' ' + s
            new_places.add(s)
    return new_places


def add_many_to_many(follow_relation, parallel, places):
    unique_activities = list(follow_relation.keys())
    unique_activities.sort()

    direct_predecessors = {}
    for activity in follow_relation.keys():
        if activity not in direct_predecessors: direct_predecessors[activity] = set()
        for src, dsts in follow_relation.items():
            if activity in dsts:
                direct_predecessors[activity].add(src)

    subsets = itertools.chain.from_iterable(
        itertools.combinations(unique_activities, r) for r in range(1, len(unique_activities) + 1))

    all_possible_io = itertools.product(subsets, subsets)
    actual_io = set()

    for pair in all_possible_io:
        src = pair[0]
        dst = pair[1]

        all_actual_successors = set()
        all_actual_predecessors = set()

        for s in src:
            for follower in follow_relation[s]:
                if s in parallel and follower in parallel[s]:
                    continue
                all_actual_successors.add(follower)

        for d in dst:
            for predecessor in direct_predecessors[d]:
                if d in parallel and predecessor in parallel[d]:
                    continue
                all_actual_predecessors.add(predecessor)

        if src == all_actual_predecessors and dst == all_actual_successors:
            actual_io.add((src, dst))

    new_places = set()
    new_places = new_places.union(places)

    for io in actual_io:
        s = ""
        t = ""

        for e in sorted(list(io[0])):
            s += e + " "
        for e in sorted(list(io[1])):
            t += e + " "
        if s.replace(' ','') != '' and t.replace(' ','') != '':
            new_places.add(s + "---> " + t[:-1])

    return new_places


def add_places(net_places, alpha_relations):
    places = set()
    places.union(net_places)
  
    eventually_follows = alpha_relations["eventually_follows"]
    parallel = alpha_relations["parallel"]

    places = add_one_to_one(eventually_follows, places)

    places = add_one_to_many(eventually_follows, parallel, places)
    places = add_many_to_one(eventually_follows, parallel, places)
    places = add_many_to_many(eventually_follows, parallel, places)

    return places


def find_actual_places(net):
    n = net.places
    net_places = set()
    start_places = set()
    end_places = set()

    def parse_element(element):
        to_skip = ['"', "'", '(', ')', '{', '}', ',', ' ']
        activities = [a for a in element if a not in to_skip]
        return sorted(activities)

    for i in n:
        if i.name != "start" and i.name != "end":
            elements = re.findall(r"({[^{}]+})", i.name)
            sources = parse_element(elements[0])
            targets = parse_element(elements[1])

            place = ' '.join(sources) + ' ---> ' + ' '.join(targets)
            net_places.add(place)

        elif i.name == "start":
            for t in i.out_arcs:
                start_places.add(t.target.name)
        elif i.name == "end":
            for s in i.in_arcs:
                end_places.add(s.source.name)
    
    sp = list(start_places)
    sp.sort()

    ep = list(end_places)
    ep.sort()

    start_place = '< ---> ' + ' '.join(sp)
    end_place = ' '.join(ep) + ' ---> |'

    net_places.add(start_place)
    net_places.add(end_place)

    return net_places


def build_net_from_places(unique_activities, places):
    net = PetriNet()
    transitions = {}
    for activity in unique_activities:
        label = None if activity == '<' or activity == '|' else activity
        t = PetriNet.Transition(activity, label)
        transitions[activity] = t
        net.transitions.add(t)
    
    i = 0
    for place in places:
        elements = place.split(' ---> ')
        sources = elements[0].split(' ')
        targets = elements[1].split(' ')
        
        p = PetriNet.Place('p'+str(i))
        net.places.add(p)
        
        for source in sources:
            petri_utils.add_arc_from_to(transitions[source], p, net)
        for target in targets:
            petri_utils.add_arc_from_to(p, transitions[target], net)
        i += 1
        
    

    source = PetriNet.Place("source")
    sink = PetriNet.Place("sink")
    net.places.add(source)
    net.places.add(sink)

    petri_utils.add_arc_from_to(source, transitions['<'], net)
    petri_utils.add_arc_from_to(transitions['|'], sink, net)
    
    im = Marking()
    fm = Marking()
    im[source] = 1
    fm[sink] = 1

    return net, im, fm


def back_to_petri(edge_index, nodes, mask):
  pn = PetriNet()  # "new_petri_net"
  pn_dict = {}
  for i in range(len(nodes)):
    if mask[i]:
      if "p" in nodes[i]:
        place = PetriNet.Place(nodes[i])
        pn_dict[i] = place
        pn.places.add(place)
      else:
        name = nodes[i] if nodes[i] != '<' and nodes[i] != '|' else None
        transition = PetriNet.Transition(nodes[i], name)
        pn_dict[i] = transition
        pn.transitions.add(transition)

  #petri_utils.add_arc_from_to(source, t_1, net)
  for i in range(len(edge_index[0])):
    if mask[edge_index[0][i].item()] == 1 and mask[edge_index[1][i].item()] == 1:
      src = pn_dict[edge_index[0][i].item()]
      dst = pn_dict[edge_index[1][i].item()]
      petri_utils.add_arc_from_to(src, dst, pn)

  im = Marking()
  source = PetriNet.Place("source")
  pn.places.add(source)
  petri_utils.add_arc_from_to(source, pn_dict[nodes.index('<')], pn)
  im[source] = 1

  fm = Marking()
  sink = PetriNet.Place("sink")
  pn.places.add(sink)
  petri_utils.add_arc_from_to(pn_dict[nodes.index('|')], sink, pn)
  fm[sink] = 1

  return pn, im, fm


