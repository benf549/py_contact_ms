import numpy as np
import prody as pr
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from py_contact_ms import get_radii_from_names, calculate_contact_ms, calculate_maximum_possible_contact_ms

import io
import os
from typing import Tuple

table = Chem.GetPeriodicTable()
rdBase.DisableLog('rdApp.*')

def calc_cms(
    pdb_file: os.PathLike, ligand_smiles: str, 
    prody_selection_protein: str = "not hetero", 
    prody_selection_ligand: str = "hetero"
) -> Tuple[float, float, float]:
    contact_ms, max_cms, norm_cms = np.nan, np.nan, np.nan
    try:
        complex_strip = pr.parsePDB(str(pdb_file)).select('not element H').copy()

        ### Compute protein features
        binder_xyz = complex_strip.select(prody_selection_protein).getCoords()
        binder_res = complex_strip.select(prody_selection_protein).getResnames()
        binder_atoms = complex_strip.select(prody_selection_protein).getNames()
        binder_radii = get_radii_from_names(binder_res, binder_atoms)

        ### Compute ligand features
        sstream = io.StringIO()
        target_xyz, target_radii = [], []
        pr.writePDBStream(sstream, complex_strip.select(prody_selection_ligand).copy())
        mol = Chem.MolFromPDBBlock(sstream.getvalue())
        smimol = Chem.MolFromSmiles(ligand_smiles)
        mol = AllChem.AssignBondOrdersFromTemplate(smimol, mol)
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            target_xyz.append([pos.x, pos.y, pos.z])
            target_radii.append(table.GetRvdw(atom.GetAtomicNum()))

        # Compute contact MS and normalize by maximum possible contact MS
        max_cms, _, _ = calculate_maximum_possible_contact_ms(target_xyz, target_radii)
        contact_ms, _, _ = calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)
        norm_cms = contact_ms / max_cms
    except Exception:
        pass
    return contact_ms, max_cms, norm_cms


def calc_pl_cms():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate contact MS for a given protein-ligand complex.")
    parser.add_argument("pdb_file", type=str, help="Path to the PDB file of the protein-ligand complex.")
    parser.add_argument("ligand_smiles", type=str, help="SMILES string of the ligand.")
    parser.add_argument("--prody_selection_protein", type=str, default="not hetero", help="ProDy selection string for the protein (default: 'not hetero').")
    parser.add_argument("--prody_selection_ligand", type=str, default="hetero", help="ProDy selection string for the ligand (default: 'hetero').")

    args = parser.parse_args()
    contact_ms, max_cms, norm_cms = calc_cms(**vars(args))
    print("contact_ms, max_cms, norm_cms")
    print(contact_ms, max_cms, norm_cms, sep=", ")


if __name__ == "__main__":
    calc_pl_cms()