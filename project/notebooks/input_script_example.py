from pathlib import Path
import sys

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

from utils.petri_net_utils import add_places, get_alpha_relations, build_net_from_places

def build_net_from_log(unique_activities, log):
    dict_alpha_relations = get_alpha_relations(log, depth=1)
    places = add_places(dict_alpha_relations, further_than_one_hop=False)
    input_net, input_im, input_fm = build_net_from_places(unique_activities, places)
    return input_net, input_im, input_fm