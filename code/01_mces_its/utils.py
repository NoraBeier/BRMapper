#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
import networkx.algorithms.isomorphism as iso

import matplotlib.pyplot as plt
import numpy as np

from synutility.SynVis.graph_visualizer import GraphVisualizer
import matplotlib.pyplot as plt

from itertools import product
from itertools import permutations

import pickle  

def parse_reactions(file_path):
    
    rxn = {}
   
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                key, smiles, rest = line.strip().split(maxsplit=2)
                rxn[key] = smiles
    return rxn

def smiles_to_graph(smiles, start_index, add_hydrogen = True):
    
    # ATOM INFORMATION
    def get_atom_features(molecule):
        atoms = molecule.GetAtoms()
        features = {}
        for atom in atoms:
            # atom type
            element = atom.GetSymbol()
            element_num = atom.GetAtomicNum()
            # Number of Heavy Neighbors
            heavy_neighbors = min(atom.GetDegree(), 5)  # 5 stands for "More than four"
            # Formal Charge
            formal_charge = atom.GetFormalCharge()
            if formal_charge < -3:
                formal_charge = -4  # "Extreme"
            elif formal_charge > 3:
                formal_charge = 4  # "Extreme"
            # Hybridization
            hybridization = atom.GetHybridization()
            hybridization_map = {
                Chem.rdchem.HybridizationType.S: 0,
                Chem.rdchem.HybridizationType.SP: 1,
                Chem.rdchem.HybridizationType.SP2: 2,
                Chem.rdchem.HybridizationType.SP3: 3,
                Chem.rdchem.HybridizationType.SP3D: 4,
                Chem.rdchem.HybridizationType.SP3D2: 5,
            }
            hybridization_value = hybridization_map.get(hybridization, 6)  # "Other"
            # Explicit Valence
            explicit_valence = min(atom.GetExplicitValence(), 6)  # Maximalwert 6
            # Is In Ring
            in_ring = int(atom.IsInRing())
            # Aromaticity
            aromatic = int(atom.GetIsAromatic())
            # Atomic Mass Scaled (normalized atom mass)
            atomic_mass = atom.GetMass() / 200.0  # Normalisierung: angenommener Maximalwert 200
            # VDW Radius Scaled (normalized van der Waals radius)
            vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 2.0  # Normalisierung: angenommener Maximalwert 2.0
            # Covalent Radius Scaled (normalized covalent radius)
            covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) / 2.0  # Normalisierung: angenommener Maximalwert 2.0
            # Chirality Type
            chirality_type = atom.GetChiralTag()
            chirality_map = {
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
            }
            chirality_value = chirality_map.get(chirality_type, 3)  # "Other"
            # Number of Hydrogen
            num_hydrogens = min(atom.GetTotalNumHs(), 5)  # 5 steht für "More than four"  
            
            # mapped in input
            map_num = atom.GetAtomMapNum()
            if not map_num:
                map_num = -1
            
            features[atom.GetIdx()] = {
                                    'element':element, 
                                    'element_num':element_num,
                                    'heavy_neighbors':heavy_neighbors,
                                    'formal_charge':formal_charge,
                                    'hybridization':hybridization_value,
                                    'explicit_valence':explicit_valence,
                                    'in_ring':in_ring,     
                                    'aromatic':aromatic,
                                    'atomic_mass':atomic_mass,
                                    'vdw_radius':vdw_radius,
                                    'covalent_radius':covalent_radius,
                                    'chirality_value':chirality_value,
                                    'num_hydrogens':num_hydrogens,
                                    'mapped': map_num
                             }
                                    
        return features 
     
     # BOND INFORMATION
    def get_bond_features(molecule):
        if molecule is None:
            raise ValueError("The molecule is invalid or could not be created.")

        bonds = molecule.GetBonds()
        bond_features = {}

        for bond in bonds:
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_type_map = {
                Chem.rdchem.BondType.SINGLE: 1,
                Chem.rdchem.BondType.DOUBLE: 2,
                Chem.rdchem.BondType.TRIPLE: 3,
                Chem.rdchem.BondType.AROMATIC: 1.5,
            }
            order = bond_type_map.get(bond_type, -1)
            conjugated = int(bond.GetIsConjugated())
            in_ring = int(bond.IsInRing())

            bond_features[(atom1_idx, atom2_idx)] = {
                "order": order,
                "conjugated": conjugated,
                "in_ring": in_ring
            }

        return bond_features     
     
    def unset_aromatic_flags(m):
        for b in m.GetBonds():
            if b.GetBondType() != Chem.BondType.AROMATIC and b.GetIsAromatic():
                b.SetIsAromatic(False)
                b.GetBeginAtom().SetIsAromatic(False)
                b.GetEndAtom().SetIsAromatic(False)
        return m
     
    def filter_unmapped(G):
        G.remove_nodes_from([n for n in G.nodes() if G.nodes[n]['mapped'] < 0]) 
     
                  
    pmol = Chem.MolFromSmiles(smiles)
    if pmol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if add_hydrogen:
        pmol = Chem.AddHs(pmol, addCoords=True)
    else:
        pmol = Chem.DeleteSubstructs(pmol, Chem.MolFromSmarts("[#1]"), onlyFrags=True)

    all_kekule = Chem.ResonanceMolSupplier(pmol, Chem.KEKULE_ALL)

    graphs = []
    for mol in all_kekule:  
        unset_aromatic_flags(mol)

        #print(Chem.MolToSmiles(mol,kekuleSmiles=True))

        features = get_atom_features(mol)
        graph = nx.Graph()
        old_to_new_id = {atom.GetIdx(): start_index + idx for idx, atom in enumerate(mol.GetAtoms())}

        for atom in mol.GetAtoms():
            new_idx = old_to_new_id[atom.GetIdx()]
            graph.add_node(new_idx, **features[atom.GetIdx()])
 
        bond_features = get_bond_features(mol)
        for bond in mol.GetBonds():
            begin_idx = old_to_new_id[bond.GetBeginAtomIdx()]
            end_idx = old_to_new_id[bond.GetEndAtomIdx()]
            graph.add_edge(begin_idx, end_idx, **bond_features[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])


        filter_unmapped(graph)

        graphs.append(graph)

    return graphs

def is_balanced(reactants_graph, products_graph):
    
    relements = []
    for a, d in reactants_graph.nodes(data=True):
        relements.append(d['element'])

    pelements = []
    for a, d in products_graph.nodes(data=True):
        pelements.append(d['element'])

    relements.sort()
    pelements.sort()

    return relements == pelements

def annotate_hydrogen_order(graph):
    
    # this function only work because hydrogen can only have one edge/bond
    hydrogen_groups = []
    for a in graph.nodes():
        group = []
        hydrogen_counter = 1
        for na in graph.neighbors(a):
            if graph.nodes[na]['element'] == "H":
                group.append((a, na))
                graph.nodes[na]['hydrogen_index'] = hydrogen_counter
                hydrogen_counter += 1

        if len(group) > 0:
            hydrogen_groups.append(group)

    forward_demands = {}
    backwards_demands = {}

    for hgroup in hydrogen_groups:
        for i in range(len(hgroup)):
            forward_demands[hgroup[i]] = hgroup[i+1:] 
            backwards_demands[hgroup[i]] = hgroup[:i] 

    return forward_demands, backwards_demands
 
def plot_its(its, filename):
    fig, ax = plt.subplots(1, 1, figsize = ( 20, 30 ))
    vis = GraphVisualizer()
    vis.plot_its(its, ax, use_edge_color = True)
    plt.savefig(filename)
    plt.close()   
    
def write_its(its, filename, seperator):
    with open(filename, "a") as f:
        f.write(">>>"+seperator+"\n")
        for n, d in its.nodes(data=True):
            f.write(str(n)+" "+str(d['element'])+"\n")
        for line in nx.generate_edgelist(its):
            f.write(line+"\n")

def write_pickle_its(its_list, filename, key):
    with open(filename+"_"+key+".pickle", 'wb') as handle:
        pickle.dump(its_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def permutation_product(a, b):
    l = len(b)

    alt_matching = []
    for p in permutations(range(l)):
        alt_matching.append( [ (a[i], b[p[i]]) for i in range(l) ] ) 
    return alt_matching

def build_its_graphs(reactants_graph, products_graph, mapping, its_graphs):

    its_base = nx.Graph()
    its_base.add_nodes_from(reactants_graph.nodes(data=True))

    # we can use educt graph directly
    for u,v, d in reactants_graph.edges(data=True): 
        its_base.add_edge(u,v, rorder=d['order'], rconjugated=d['conjugated'], rin_ring=d['in_ring'])

    # mark all nodes by the adjacent mappings, afterwards all nodes with the same signature in the mapping can be actually matched!
    educt_signature = {}
    product_signature = {}
    educt_matched_nodes = set()
    product_matched_nodes = set()
    for j, ((us, ut) , (vs, vt)) in enumerate(mapping):

        educt_matched_nodes.add(us)
        educt_matched_nodes.add(ut)
        product_matched_nodes.add(vs)
        product_matched_nodes.add(vt)

        educt_signature.setdefault(us, set()).add(j)
        educt_signature.setdefault(ut, set()).add(j)
        product_signature.setdefault(vs, set()).add(j)
        product_signature.setdefault(vt, set()).add(j)
                
    # now group nodes by element and signature
    inv_educt_signature = {}
    for node in educt_signature:
        inv_educt_signature.setdefault( ( reactants_graph.nodes[node]['element'], frozenset(educt_signature[node]) ), []).append(node)

    inv_product_signature = {}
    for node in product_signature:
        inv_product_signature.setdefault( ( products_graph.nodes[node]['element'], frozenset(product_signature[node]) ), []).append(node)    
    
    # find unmatched
    unmatched_educt_nodes = set(reactants_graph.nodes()).difference(educt_matched_nodes)
    unmatched_product_nodes = set(products_graph.nodes()).difference(product_matched_nodes)
    
    # add unmatched to signature
    for node in unmatched_educt_nodes:
        inv_educt_signature.setdefault( ( reactants_graph.nodes[node]['element'], frozenset([]) ), []).append(node)
    for node in unmatched_product_nodes:
        inv_product_signature.setdefault( ( products_graph.nodes[node]['element'], frozenset([]) ), []).append(node)    

    #print(inv_educt_signature)
    #print(inv_product_signature)

    # now do the combinatorics... by definition, both signatures have the same keyset
    local_combinations = []
    for signature in inv_educt_signature:
        local_combinations.append( permutation_product(inv_educt_signature[signature], inv_product_signature[signature]) )

    global_combinations = list(product(*local_combinations))   # never forget flattening

    if len(global_combinations) > 50:
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

def find_isomorphism_class(reaction, cluster_space):
    """
    Attempts to find the isomorphism class of a reaction.
    Returns the index of the cluster if found, otherwise None.
    """
    for i, cluster in enumerate(cluster_space):
        if nx.is_isomorphic(
            cluster, reaction,
            node_match=iso.categorical_node_match(["charge", "element"], [1, 'H']),
            edge_match=iso.categorical_edge_match("order", (1.0, 1.0))
        ):
            return i
    return None

def filter_its_graphs(its_graphs) :
    """
    Groups reactions into isomorphism classes.
    Returns the cluster space.
    """
    cluster_space = []
    for reaction in its_graphs:
        index = find_isomorphism_class(reaction, cluster_space)
        if index is None:
            cluster_space.append(reaction)
            
    return cluster_space

def transform2line_graph(graph):
    line = nx.line_graph(graph)
    # labels for new nodes (which represend the bonds of the molecule)
    for edge in line:
        u, v = edge 
        edge_label = graph.edges[edge].get('order') 
        u_element = graph.nodes[u].get('element')
        u_hydro = graph.nodes[u].get('hydrogen_index', 0)
        v_element = graph.nodes[v].get('element')
        v_hydro = graph.nodes[v].get('hydrogen_index', 0)
        elements_set = frozenset({u_element, v_element})
        line.nodes[edge]['label'] = (elements_set, edge_label)
        line.nodes[edge]['hydrogen_index'] = max(u_hydro, v_hydro) 

    # labels for new edges (which represend the atoms of the molecule)
    for atom in line.edges():
        # find the right atom
        if atom[0][0] == atom[1][0] or atom[0][0] == atom[1][1]:        
            conacted = atom[0][0]   
        elif atom[0][1] == atom[1][0] or atom[0][1] == atom[1][1]:
            conacted = atom[0][1]
        else:
            print('Error in Line Edge Labels')
            exit()   
        attributes = graph.nodes[conacted]
        for key, value in attributes.items():
            if key == 'element':
                line.edges[atom]['label'] = value
            else:
                line.edges[atom][key] = value
                                
    return line


def labeled_modular_product(G, H):
    
    M = nx.Graph()
    
    # Matching of Nodes and Edges
    def node_match(attr1, attr2):
        return attr1["label"] == attr2["label"] and ('hydrogen_index' not in attr1 or attr1['hydrogen_index'] >= attr2['hydrogen_index'])
    def edge_match(attr1, attr2):
        return attr1["label"] == attr2["label"]
    
    # Create all possible node pairs
    for (x, l1), (u, l2) in product(G.nodes(data=True), H.nodes(data=True)):
        if node_match(l1, l2):  
            M.add_node((x, u))

    # Create all possible edge pairs
    for (g1,h1),(g2,h2) in product(M.nodes(),M.nodes()):
        #print("Check Edge", (g1,h1),(g2,h2) )
        if g1 == g2 or h1 == h2: # exclude all loops
            continue    
            
        if G.has_edge(g1,g2) == H.has_edge(h1,h2): 
            if G.has_edge(g1,g2):
                if edge_match(G.edges[(g1,g2)],H.edges[(h1,h2)]):
                     M.add_edge((g1, h1), (g2, h2))     
            else:
                M.add_edge((g1, h1), (g2, h2))
    return M

    
    

