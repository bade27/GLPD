from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


from itertools import combinations, product
import re
import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from utils.pm4py_utils import get_variants



def alpha_relations(log):
    variants_count = get_variants(log)

    directly_follows = {}
    for variant in variants_count:
        activities = variant["variant"].split(',')
        activities.insert(0, '<')
        activities.append('|')
        for i in range(len(activities)-1):
            if activities[i] not in directly_follows:
                directly_follows[activities[i]] = set()
            directly_follows[activities[i]].add(activities[i+1])
        if activities[-1] not in directly_follows:
            directly_follows[activities[-1]] = set()

    causality = lambda x, y: (y in directly_follows[x]) and not (x in directly_follows[y])
    parallel = lambda x, y: (y in directly_follows[x]) and (x in directly_follows[y])
    unrelated = lambda x, y: not (y in directly_follows[x]) and not (x in directly_follows[y])

    unique_activities = directly_follows.keys()

    relations = []
    for x in unique_activities:
        for y in unique_activities:
            if causality(x, y):
                if [x, y, "->"] not in relations:
                    relations.append([x, y, "->"])
                if [y, x, "<-"] not in relations:    
                    relations.append([y, x, "<-"])
            elif parallel(x, y):
                if [x, y, "//"] not in relations:
                    relations.append([x, y, "//"])
                if [y, x, "//"] not in relations:
                    relations.append([y, x, "//"])
            elif unrelated(x, y):
                relations.append([x, y, "#"])

    df = pd.DataFrame(relations, columns=['X', 'Y','Z'])

    df = df.pivot(index='X', columns='Y', values='Z').reset_index()

    df.set_index("X", inplace=True)

    return df


def add_dfg_rel(parent_of, places):
    new_places = set()
    new_places = new_places.union(places)
    for key, element in parent_of.items():
        for val in sorted(list(element)):
            s = key + " ---> "
            s += val + " "
            new_places.add(s[:-1])
            s = ""

    return new_places


def add_many_to_one_many_to_many(parent_of, parallel, places):
    unique_activities = set(parent_of.keys()).difference(('<','|'))

    unique_activities = list(unique_activities)
    unique_activities.sort()

    common_children = {}
    for i in range(len(unique_activities)):
        for j in range(len(unique_activities)):
            if i != j:
                if i in parallel and j in parallel[i]:
                    continue
                intersection = tuple(parent_of[unique_activities[i]].intersection(parent_of[unique_activities[j]]))
                if '|' in intersection:
                    idx = intersection.index('|')
                    intersection = intersection[:idx] + intersection[idx+1:]
                
                if len(intersection) > 0:
                    if intersection not in common_children:
                        common_children[intersection] = set()
                    common_children[intersection].add(unique_activities[i])
                    common_children[intersection].add(unique_activities[j])

    new_places = set()
    new_places = new_places.union(places)

    for dst, src in common_children.items():
        output_combinations = []
        input_combinations = []
        for n in range(1, len(dst)+1):
            comb = combinations(list(dst),n)
            for c in comb:
                output_combinations.append(c)
        for n in range(1, len(src)+1):
            comb = combinations(list(src),n)
            for c in comb:
                input_combinations.append(c)

        cross_prod = product(input_combinations, output_combinations)
        for cp in cross_prod:
            s = ""
            t = ""

            for e in sorted(list(cp[0])):
                s += e + " "
            for e in sorted(list(cp[1])):
                t += e + " "
            if s.replace(' ','') != '' and t.replace(' ','') != '':
                new_places.add(s + "---> " + t[:-1])

    return new_places


def add_one_to_many(parent_of, parallel, places):
    new_places = set()
    new_places = new_places.union(places)
    
    for key, element in parent_of.items():
        possible_outgoing_places = []
        all_children = sorted(list(element))

        # remove parallel from list
        not_to_consider = set()
        for child1 in all_children:
            for child2 in all_children:
                if child1 in parallel and child2 in parallel[child1]:
                    not_to_consider.add(child1)
                    not_to_consider.add(child2)
                    
        all_children = sorted(list(element.difference(not_to_consider)))
        
        for n in range(1, len(all_children)+1):
            comb = combinations(all_children, n)
            for c in comb:
                possible_outgoing_places.append(c)
        cross_prod = product([key], possible_outgoing_places)

        for cp in cross_prod:
            s = cp[0] + ' ---> '
            for x in cp[1]:
                s += x + ' '
            new_places.add(s[:-1])
    return new_places


def add_places(dataframe, relations, net_places=None):
    places = set()

    if net_places:
        places.union(net_places)
    
    causality = {}
    parallel = {}

    for _, row in dataframe.iterrows():
        if relations.loc[row["source"],row["destination"]] == "->":
            if row["source"] not in causality:
                causality[row["source"]] = set()
            causality[row["source"]].add(row["destination"])
        elif relations.loc[row["source"], row["destination"]] == "//":
            if row["source"] not in parallel:
                parallel[row["source"]] = set()
            parallel[row["source"]].add(row["destination"])

    places = add_dfg_rel(causality, places)

    place_start_end = set()
    for place in places:
        if '<' in place or '|' in place:
            place_start_end.add(place)
    places = places.difference(place_start_end)

    places = add_one_to_many(causality, parallel, places)
    places = add_many_to_one_many_to_many(causality, parallel, places)

    return places.union(place_start_end)


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


