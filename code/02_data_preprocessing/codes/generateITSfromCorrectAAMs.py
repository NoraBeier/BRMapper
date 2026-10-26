import os
import csv
import sys
import ast
import pickle
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from synkit.IO import rsmi_to_its, smart_to_gml,gml_to_its
from synkit.Vis import GraphVisualizer
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

# parse the Attributes from MCSplit Results to the Synkit Atrributs to do a isomorpic check 
def transform_graph_attributes(G):
    for node, attrs in G.nodes(data=True):
        if 'formal_charge' in attrs:
            G.nodes[node]['charge'] = attrs.pop('formal_charge')
    for u, v, attrs in G.edges(data=True):
        if 'order' in attrs:
            val = attrs['order']
            if isinstance(val, tuple) and all(isinstance(x, (int, float)) for x in val):
                cleaned = tuple(float(x) for x in val)
                G.edges[u, v]['order'] = cleaned
    return G

# Find Correct Atom-to- Atom Map from the Suboptimals
aams = {}
results = {}
no_corr_found = {}
case_counter = 0
neg_counter = 0
corr_bigger = 0
brmapper_bigger = 0
path_rxns = sys.argv[1]
with open(path_rxns,'r') as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
        aam = line.split(';')
        aams[aam[0]] = aam[2]
    for maps in tqdm(aams.keys(), desc="Find Corract ITS", unit="reactions"):   
        case_counter += 1        
        # genarte ITS from corrcect Atom-to-Atom Map
        list_corr_maps = []     
        aams[maps] = aams[maps].split(',')   
        for m in aams[maps]:
            gml = smart_to_gml(m, sanitize=False, core=False, explicit_hydrogen=False)
            corr_its = gml_to_its(gml)
            # remove [+H] from corr_its            
            nodes_to_remove = [n for n, attrs in corr_its.nodes(data=True) if attrs.get('element') == '*']
            corr_its.remove_nodes_from(nodes_to_remove)
            list_corr_maps.append(corr_its)

        # load Suboptimals
        path_p = sys.argv[2]
        pickles_path = path_p+'/rnx_'+maps+'.pickle'
        if os.path.exists(pickles_path):
            with open(pickles_path , 'rb') as f:
                its_list = pickle.load(f)
        else:
            continue
            
                
        #find correct Atom-to-Atom Map
        found = False
        counter = 0
        corr_check = False
        brmapper_check = False
        wrong_its = set()
        if maps != 'None':       
          for its in its_list:   
            # remove all hydrogens for test
            nodes_to_remove = [n for n, attrs in its.nodes(data=True) if attrs.get('element') == 'H']
            its.remove_nodes_from(nodes_to_remove)
                
            # check for correct ITS 
            for corr_its in list_corr_maps:
                if len(corr_its.nodes()) != len(its.nodes()):
                    if len(corr_its.nodes()) > len(its.nodes()):
                        corr_check = True
                    else:
                        brmapper_check = True
                    wrong_its.add(its)
                    continue
                if len(corr_its.edges()) != len(its.edges()):
                    if len(corr_its.edges()) > len(its.edges()):
                        corr_check = True
                    else:                                   
                        brmapper_check = True
                    wrong_its.add(its)
                    continue                
                its = transform_graph_attributes(its)
                        
                # Initialize the matcher, comparing element, charge, and bond order
                gm = GraphMatcherEngine(
                    backend='nx',
                    node_attrs=['element'],
                    edge_attrs=['order']
                )
                # Test isomorphism
                are_isomorphic = gm.isomorphic(corr_its, its)
                if are_isomorphic == True:
                    if maps not in results.keys():
                        results[maps] = {} 
                        results[maps]['corr'] = counter
                    elif maps in results.keys() and 'multi' in results[maps]:
                        results[maps]['multi'].append(counter)    
                    elif maps in results.keys() and 'multi' not in results[maps]:                        
                        results[maps]['multi'] = [counter]
                    found = True
                    break
                else:
                    wrong_its.add(its)
            counter += 1
      
        if found == False:
            results[maps] = None
            neg_counter += 1
            if corr_check == True:
                corr_bigger += 1
                no_corr_found[maps] = 'Correct ITS is bigger'
            if brmapper_check == True:
                brmapper_bigger += 1                
                no_corr_found[maps] = 'brmapper ITS is bigger'

            
filtered_dict = {k: v for k, v in results.items() if v is not None}
# write csv output file
output_path = sys.argv[3]
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';') 
    writer.writerow(["RXN", "correct ITS Position","Multi Solutions"]) 
    for key, value in results.items():
        if 'multi' in value:
            writer.writerow([key, value['corr'], value['multi']])
        else: 
            writer.writerow([key, value['corr']])
print('################# Results ##################')
print('No. of Reactions:',case_counter)
print('No. of found correct in Suboptimals:',len(filtered_dict.keys()))



