#!/usr/bin/env python3

import sys

from utils import * 
import networkx as nx
from mcsplit import MCSplitWrapper, MCSplitMode

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)   
  
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

# Initialize the unified wrapper
wrapper = MCSplitWrapper(
    node_label_attr="element",
    edge_label_attr="order",
    connected=False,
)

suboptimal_cutoff = 1

def build_its_graphs_from_node_mapping(reactants_graph, products_graph, mapping, its_graphs):

    its_base = nx.Graph()
    its_base.add_nodes_from(reactants_graph.nodes(data=True))

    # we can use educt graph directly
    for u,v, d in reactants_graph.edges(data=True): 
        its_base.add_edge(u,v, rorder=d['order'], rconjugated=d['conjugated'], rin_ring=d['in_ring'])

    inv_educt_signature = {}
    inv_product_signature = {}

    educt_matched_nodes = set()
    product_matched_nodes = set()
    for i, (v1, v2) in enumerate(mapping):
        educt_matched_nodes.add(v1)
        product_matched_nodes.add(v2)

        inv_educt_signature.setdefault( ( reactants_graph.nodes[v1]['element'], frozenset([i]) ), []).append(v1)
        inv_product_signature.setdefault( ( products_graph.nodes[v2]['element'], frozenset([i]) ), []).append(v2)    
    
    # find unmatched
    unmatched_educt_nodes = set(reactants_graph.nodes()).difference(educt_matched_nodes)
    unmatched_product_nodes = set(products_graph.nodes()).difference(product_matched_nodes)
    
    # add unmatched to signature
    for node in unmatched_educt_nodes:
        inv_educt_signature.setdefault( ( reactants_graph.nodes[node]['element'], frozenset([]) ), []).append(node)

    for node in unmatched_product_nodes:
        inv_product_signature.setdefault( ( products_graph.nodes[node]['element'], frozenset([]) ), []).append(node)    

    # now do the combinatorics... by definition, both signatures have the same keyset
    local_combinations = []
    for signature in inv_educt_signature:
        local_combinations.append( permutation_product(inv_educt_signature[signature], inv_product_signature[signature]) )

    global_combinations = list(product(*local_combinations))   # never forget flattening

    if len(global_combinations) > 200:
        return

    for j, comb in enumerate(global_combinations):
        flat_comb = [x for xs in comb for x in xs ]

        pairmap = {}
        for pair in flat_comb:
            pairmap[pair[1]] = pair[0]
        
        its = its_base.copy()
        hcounter = len(its.nodes())
        for x, y in flat_comb:
            rh = reactants_graph.nodes[x]['num_hydrogens']
            ph = products_graph.nodes[y]['num_hydrogens']
            its.nodes[x]['num_hydrogens'] = 0         

            common_h = min(rh, ph)
            for i in range(common_h):
                its.add_node(hcounter, element='H', element_num=1, heavy_neighbors=1, formal_charge=0, hybridization=6, explicit_valence=1, in_ring=0, aromatic=0, atomic_mass=0.00504, vdw_radius=0.6, covalent_radius=0.155, chirality_value=0, num_hydrogens=0)
                its.add_edge(hcounter, x, porder=1, rorder=1)
                hcounter += 1
            for i in range(max(0, rh - ph)):
                its.add_node(hcounter, element='H', element_num=1, heavy_neighbors=1, formal_charge=0, hybridization=6, explicit_valence=1, in_ring=0, aromatic=0, atomic_mass=0.00504, vdw_radius=0.6, covalent_radius=0.155, chirality_value=0, num_hydrogens=0)
                its.add_edge(hcounter, x, porder=0, rorder=1)
                hcounter += 1
            for i in range(max(0, ph - rh)):
                its.add_node(hcounter, element='H', element_num=1, heavy_neighbors=1, formal_charge=0, hybridization=6, explicit_valence=1, in_ring=0, aromatic=0, atomic_mass=0.00504, vdw_radius=0.6, covalent_radius=0.155, chirality_value=0, num_hydrogens=0)
                its.add_edge(hcounter, x, porder=1, rorder=0)
                hcounter += 1

        for u, v, d in products_graph.edges(data=True): 
            if its.has_edge(pairmap[u], pairmap[v]):
                its.edges[pairmap[u], pairmap[v]]['porder']=d['order']
                its.edges[pairmap[u], pairmap[v]]['pconjugated']=d['conjugated']
                its.edges[pairmap[u], pairmap[v]]['pin_ring']=d['in_ring']
            else:
                its.add_edge( pairmap[u], pairmap[v], porder=d['order'], pconjugated=d['conjugated'], pin_ring=d['in_ring'] )

        # clean up the its, create pair order
        for u,v, d in its.edges(data=True): 
            order = ( d.get('rorder', 0), d.get('porder', 0) )
            conjugated = ( d.get('rconjugated', 0), d.get('pconjugated', 0) )
            in_ring = ( d.get('rin_ring', 0), d.get('pin_ring', 0) )
            its.edges[u,v].clear()

            its.edges[u,v]['order'] = order
            its.edges[u,v]['conjugated'] = conjugated
            its.edges[u,v]['in_ring'] = in_ring

        its_graphs.append(its)


def process_reactions(key, reaction):
    try:
        #print("============", key)
        reactants_smiles, catalyst, products_smiles = reaction.split('>')
        #print(reactants_smiles, catalyst, products_smiles)

        # create reactant graph
        #print('reactant:',reactants_smiles)
        reactants_graphs = smiles_to_graph(reactants_smiles, 0, add_hydrogen = False)

        # create prodoct graph
        #print('products:',products_smiles)
        begin_ID = len(reactants_graphs[0].nodes)
        products_graphs = smiles_to_graph(products_smiles, begin_ID, add_hydrogen = False)

        #print("RGs", len(reactants_graphs), "PGs", len(products_graphs))

    except Exception as e:
        print(f"Error by reaction {key}: {e}")
        return "", []


    all_cluster_its = []
    for reactants_graph in reactants_graphs:
      for products_graph in products_graphs:
        
        if not is_balanced(reactants_graph, products_graph):
            print("Unbalanced Reaction", key)
            return "", []

        # create the suboptimal atom-to-atom maps of the line graphs    
        result_mces = wrapper.solve(reactants_graph, products_graph, mode=MCSplitMode.LINE_SUBOPT)

     
        if result_mces.size > 200:
            return "", []
            
        its_graphs = []
        for i, mapping in enumerate(result_mces.solutions):
            build_its_graphs_from_node_mapping(reactants_graph, products_graph, mapping['atom_map'].items(), its_graphs)
            
        all_cluster_its.extend(its_graphs)
 
    if len(all_cluster_its) > 200:
        return "", [] 
        
    return key, all_cluster_its


if __name__ == "__main__":

    reaction = sys.argv[1]
    outputfolder = sys.argv[2]

    key, smiles = reaction.strip().split()

    key, its_graphs = process_reactions(key, smiles)

    if len(its_graphs) > 0:
        write_pickle_its(its_graphs, outputfolder+"/rxn", key)

