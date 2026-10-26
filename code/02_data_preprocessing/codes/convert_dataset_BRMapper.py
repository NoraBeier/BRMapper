import os
import csv
import ast
import sys
import pickle
import networkx as nx
from tqdm import tqdm
from TUDataset import TUDataset
from data_utils import its_to_TUData
import matplotlib.pyplot as plt

def convert_node_attributes_to_features(graph):

    attribute_keys = [
        'element_num',
        'heavy_neighbors',
        'formal_charge',
        'hybridization',
        'explicit_valence',
        'in_ring',
        'aromatic',
        'atomic_mass',
        'vdw_radius',
        'covalent_radius',
        'chirality_value',
        'num_hydrogens',
        'hydrogen_index'
    ]

    for node in graph.nodes:
        attrs = graph.nodes[node]
        features = [attrs.get(key, 0) for key in attribute_keys]
        graph.nodes[node].clear()
        graph.nodes[node]['node_features'] = features

    attribute_keys = [
        'order',
        'conjugated',
        'in_ring',
    ]
    
    for edge in graph.edges:
        attrs = graph.edges[edge]
        features = [attrs.get(key, 0) for key in attribute_keys]
        graph.edges[edge].clear()
        #graph.edges[edge]['edge_features'] = features
        graph.edges[edge]['edge_features'] = [x for tup in features for x in tup] #version without tubles
    return graph
    
def generate_GT(len_r, corr):
    if not (0 <= corr < len_r):
        print(corr,len_r)
        raise ValueError("Input Error.")
    result = [0] * len_r
    result[corr] = 1
    return result

# Convert the its graph to the TUDataset format. We set the target label to 1 and the rest with label 0
dataset_name = sys.argv[1]
dataset_dir = sys.argv[2]

# load Ground Truth
filename = sys.argv[4]
rxn_id = {}
csv.field_size_limit(sys.maxsize)

with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile,delimiter=';')
    rows = list(reader)
    for row in rows:
        if not row[1]:
            continue
        # Filter Reactions with to many possitives
        if len(row) == 3:
            if len(row[2]) > 1000:
                continue
        rxn_id[row[0]] = {} 
        rxn_id[row[0]]['corr'] = row[1]
        if len(row) == 3:                
            rxn_id[row[0]]['filter'] = row[2]
        else:
            rxn_id[row[0]]['filter'] = 'None'

count = 0    
ids = []
new_data = []
gt_all =[]
id_file = dataset_dir + "/" + dataset_name + "/" + dataset_name + "_ids.txt"
if os.path.exists(id_file):
    os.remove(id_file)
    open(id_file, "w").close()
else:
    os.makedirs(os.path.dirname(id_file), exist_ok=True)

# load subpotimals
for no in rxn_id.keys():
    empty = False
    if no == 'RXN':
        continue
    corr = int(rxn_id[no]['corr'])
    if rxn_id[no]['filter'] != 'None':
        filter_rxn = ast.literal_eval(rxn_id[no]['filter'])
    else: 
        filter_rxn = []
    print('Process Reaction',no,'with Ground Truth Position:',corr, 'and Filter reactions:',rxn_id[no]['filter'])    
    path = sys.argv[3]    
    with open(path+'/rnx_'+no+'.pickle', 'rb') as f:
        its_list = pickle.load(f)
    print('No. of Suboptimals',len(its_list))
    if len(its_list) > 400:
        continue
    count = count+1
    
    # convert Features
    _ids = []
    its_list_mod = []
    filter_count = 0
    for its in its_list:
        if filter_count in filter_rxn:
            filter_count = filter_count + 1
            continue
        filter_count = filter_count + 1
        if its.number_of_nodes() == 0:
            empty = True
            break
        its_mod = convert_node_attributes_to_features(its)
        its_list_mod.append(its_mod)
        _ids.append(str(no))
    if empty:
        continue
    new_data = new_data + its_list_mod
    ids = ids + _ids
    gt = generate_GT(len(its_list_mod), corr)
    gt_all = gt_all + gt

its_to_TUData(new_data, gt_all , dataset_name, dataset_dir)
with open(id_file, "w") as f:
    f.writelines([id + "\n" for id in ids])
    
# Load the TUDataset. This dataset is a torch.utils.data.Dataset
dataset = TUDataset(dataset_dir, dataset_name, use_node_attr=True, use_edge_attr=True, force_reload=True)
        
        
