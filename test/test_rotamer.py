import os
import sys
import numpy as np

import torch

from torchdrug import data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from diffpack import rotamer


if __name__ == "__main__":
    protein = data.Protein.from_pdb('10mh_A.pdb', atom_feature=None, bond_feature=None,
                                    residue_feature=None, mol_feature=None)
    protein = protein.subgraph(protein.atom_name != 37)


    chis = rotamer.get_chis(protein)
    rotate_angles = torch.zeros_like(chis)
    new_protein = protein.clone()
    new_protein = rotamer.rotate_side_chain(protein, rotate_angles)
    assert (new_protein.node_position == protein.node_position).all()

    for i in range(8):
        new_protein = protein.clone()
        rotamer.rotate_side_chain(new_protein, rotate_angles)
        new_chis = rotamer.get_chis(new_protein)
        diff = (new_chis - chis).fmod(np.pi * 2)
        test_mask = diff.isnan() | ((diff - np.pi * i / 4).abs() < 1e-4) | ((diff + np.pi * (8-i) / 4).abs() < 1e-4)
        if not test_mask.all():
            import pdb; pdb.set_trace()
        rotate_angles = rotate_angles + np.pi / 4 
