import csv
import itertools
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def extract_products_from_rxn_smiles(rxn_smiles: str):
    try:
        _, products_part = rxn_smiles.split(">>")
    except ValueError:
        raise ValueError("Ungültiger Reaction-SMILES – kein '>>' gefunden.")
    # Produkte können durch '.' getrennt sein
    product_smiles = [s.strip() for s in products_part.split('.') if s.strip()]
    return product_smiles


def enumerate_tautomers(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    te = rdMolStandardize.TautomerEnumerator()
    tautomers = te.Enumerate(mol)
    unique = set()
    for t in tautomers:
        try:
            Chem.SanitizeMol(t)
            smi = Chem.MolToSmiles(Chem.RemoveHs(t), canonical=True)
            unique.add(smi)
        except Exception:
            pass
    return sorted(unique)

def generate_tautomeric_reactions(rxn_smiles: str):
    try:
        reactants_part, products_part = rxn_smiles.split(">>")
    except ValueError:
        return [rxn_smiles]  # kein valides Reaction-SMILES

    product_smiles = [s.strip() for s in products_part.split('.') if s.strip()]
    if not product_smiles:
        return [rxn_smiles]

    tautomer_lists = [enumerate_tautomers(smi) or [smi] for smi in product_smiles]

    from itertools import product
    all_combinations = list(product(*tautomer_lists))
    tautomeric_rxns = []
    for comb in all_combinations:
        new_products = '.'.join(comb)
        tautomeric_rxns.append(f"{reactants_part}>>{new_products}")
    if rxn_smiles not in tautomeric_rxns:
        tautomeric_rxns.append(rxn_smiles)

    return tautomeric_rxns

def unset_aromatic_flags(m):
         for b in m.GetBonds():
             if b.GetBondType() != Chem.BondType.AROMATIC and b.GetIsAromatic():
                 b.SetIsAromatic(False)
                 b.GetBeginAtom().SetIsAromatic(False)
                 b.GetEndAtom().SetIsAromatic(False)
         return m

def main():
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    print("Lese CSV:", input_csv)

    all_rows = []

    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # erste Zeile = Header
        rows = list(reader)

    if len(header) < 3:
        raise ValueError("CSV hat weniger als 3 Spalten!")

    for row in rows:
        rxn_smiles = str(row[2]).strip()
        if not rxn_smiles or rxn_smiles.lower() == "nan":
            all_rows.append(row)
            continue

        new_tautomeric_rxns = []
        tautomeric_rxns = generate_tautomeric_reactions(rxn_smiles)

        for smiles in tautomeric_rxns:

            reaction = smiles.split('>>')

            new_educts = []
            educts = reaction[0].split('.')
            for edu in educts:
                edu = Chem.MolFromSmiles(edu)
                all_kekule = Chem.ResonanceMolSupplier(edu, Chem.KEKULE_ALL)
                kek_edu = []
                for edu2 in all_kekule:
                    edu2 = unset_aromatic_flags(edu2)
                    edu2 = Chem.MolToSmiles(edu2, kekuleSmiles=True)
                    kek_edu.append(edu2)
                new_educts.append(kek_edu)

            new_products = []
            products = reaction[1].split('.')
            for pro in products:
                pro = Chem.MolFromSmiles(pro)
                all_kekule = Chem.ResonanceMolSupplier(pro, Chem.KEKULE_ALL)
                kek_pro = []
                for pro2 in all_kekule:
                    pro2 = unset_aromatic_flags(pro2)
                    pro2 = Chem.MolToSmiles(pro2, kekuleSmiles=True)
                    kek_pro.append(pro2)
                new_products.append(kek_pro)

            all_kekule_results = {}

            if len(new_educts) > 1:
                new_educts = [list(c) for c in itertools.product(*new_educts)]
            else:
                new_educts = new_educts[0]

            if len(new_products) > 1:
                new_products = [list(c) for c in itertools.product(*new_products)]
            else:
                new_products = new_products[0]

            count = 0
            for a, b in itertools.product(new_educts, new_products):
                if not isinstance(a, list): a = [a]
                if not isinstance(b, list): b = [b]
                all_kekule_results[count] = {'educts': a, 'products': b}
                count += 1


            for k in all_kekule_results:
                ed = all_kekule_results[k]['educts']
                pr = all_kekule_results[k]['products']

                ed_str = ".".join(ed)
                pr_str = ".".join(pr)

                reaction = f"{ed_str}>>{pr_str}"
                new_tautomeric_rxns.append(reaction)

        row[2] = ",".join(new_tautomeric_rxns)
        all_rows.append(row)


    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        writer.writerows(all_rows)

    print("Fertig! Output gespeichert in:", output_csv)


if __name__ == "__main__":
    main()
