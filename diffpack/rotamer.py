import math
import numpy as np

import torch
from torchdrug.data import Protein

from diffpack.util import rot_matmul, rot_vec_mul

atom_name_vocab = {
    "C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
    "OH": 33, "OXT": 34, "SD": 35, "SG": 36
}

bb_atom_name = [atom_name_vocab[_] for _ in ['C', 'CA', 'N', 'O']]

residue_list = [
    "GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
    "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"
]
residue_vocab = {r: i for i, r in enumerate(residue_list)}
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}
one_to_three = {v: k for k, v in three_to_one.items()}
# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace

# Angle Symmetry
chi_pi_periodic_dict = {
    "ALA": [False, False, False, False],  # ALA
    "ARG": [False, False, False, False],  # ARG
    "ASN": [False, False, False, False],  # ASN
    "ASP": [False, True, False, False],  # ASP
    "CYS": [False, False, False, False],  # CYS
    "GLN": [False, False, False, False],  # GLN
    "GLU": [False, False, True, False],  # GLU
    "GLY": [False, False, False, False],  # GLY
    "HIS": [False, False, False, False],  # HIS
    "ILE": [False, False, False, False],  # ILE
    "LEU": [False, False, False, False],  # LEU
    "LYS": [False, False, False, False],  # LYS
    "MET": [False, False, False, False],  # MET
    "PHE": [False, True, False, False],  # PHE
    "PRO": [False, False, False, False],  # PRO
    "SER": [False, False, False, False],  # SER
    "THR": [False, False, False, False],  # THR
    "TRP": [False, False, False, False],  # TRP
    "TYR": [False, True, False, False],  # TYR
    "VAL": [False, False, False, False],  # VAL
}

chi_pi_periodic = [chi_pi_periodic_dict[res_name] for res_name in residue_list]

# Atom Symmetry
symm_sc_res_atoms = {
    "ARG": [["NH1", "NH2"], ["OXT", "OXT"]],  # ARG *
    "HIS": [["ND1", "CD2"], ["NE2", "CE1"]],  # HIS * *
    "ASP": [["OD1", "OD2"], ["OXT", "OXT"]],  # ASP *
    "PHE": [["CD1", "CD2"], ["CE1", "CE2"]],  # PHE * *
    "GLN": [["OE1", "NE2"], ["OXT", "OXT"]],  # GLN - check
    "GLU": [["OE1", "OE2"], ["OXT", "OXT"]],  # GLU *
    "LEU": [["CD1", "CD2"], ["OXT", "OXT"]],  # LEU - check
    "ASN": [["OD1", "ND2"], ["OXT", "OXT"]],  # ASN - check
    "TYR": [["CD1", "CD2"], ["CE1", "CE2"]],  # TYR * *
    "VAL": [["CG1", "CG2"], ["OXT", "OXT"]],  # VAL - check
}
res_sym_atom_posn = -torch.ones(len(residue_list), 2, 2, dtype=torch.long)
for res, [[a, b], [c, d]] in symm_sc_res_atoms.items():
    res_sym_atom_posn[Protein.residue2id[res]] = torch.tensor([
        [atom_name_vocab[c], atom_name_vocab[d]],
        [atom_name_vocab[a], atom_name_vocab[b]]
    ])

restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

rigid_group_atom_positions = {
    # Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
    # psi and chi angles:
    # 0: 'backbone group',
    # 1: 'pre-omega-group', (empty)
    # 2: 'phi-group', (currently empty, because it defines only hydrogens)
    # 3: 'psi-group',
    # 4,5,6,7: 'chi1,2,3,4-group'
    # The atom positions are relative to the axis-end-atom of the corresponding
    # rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
    # is defined such that the dihedral-angle-definiting atom (the last entry in
    # chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
    # format: [atomname, group_idx, rel_position]
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}

"""
restype_atomname_index_map[i_resi][j_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_atom: atom name, 0-36
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
restype_atom14_index_map = -torch.ones((len(residue_list), 37), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    for value, name in enumerate(restype_name_to_atom14_names[resi_name3]):
        if name in atom_name_vocab:
            restype_atom14_index_map[i_resi][atom_name_vocab[name]] = value

"""
restype_atom14_mask[i_resi][j_atom]:
    i_resi: index of residue_list, specifies resi_type
    j_atom: atom name, 0-13
    value: 1 if atom is present in residue type, 0 otherwise
"""
"""
restype_atom14_rigid_group_positions[i_resi][j_atom][k]:
    i_resi: index of residue_list, specifies resi_type
    j_atom: atom name, 0-13
    k: 0, 1, 2  -> x, y, z coordinate
"""
restype_atom14_mask = torch.zeros((21, 14), dtype=torch.float32)
restype_atom14_rigid_group_positions = torch.zeros((21, 14, 3), dtype=torch.float32)
for i_resi, resi_name3 in enumerate(residue_list):
    for atomname, group_idx, atom_position in rigid_group_atom_positions[resi_name3]:
        atom14idx = restype_name_to_atom14_names[resi_name3].index(atomname)
        restype_atom14_mask[i_resi, atom14idx] = 1
        restype_atom14_rigid_group_positions[i_resi, atom14idx, :] = torch.as_tensor(atom_position)

chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

"""
chi_atom_index_map[i_resi][j_chi][k_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_chi: chi number, 0-3
  k_atom: k-th atom in the torsion angle, 0-3
  value: index of atom_names, specifies atom_type, -1 means no such torsion
chi_atom14_index_map[i_resi][j_chi][k_atom]:
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
chi_atom37_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
chi_atom14_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    chi_angles_atoms_i = chi_angles_atoms[resi_name3]
    for j_chi, atoms in enumerate(chi_angles_atoms_i):
        for k_atom, atom in enumerate(atoms):
            chi_atom37_index_map[i_resi][j_chi][k_atom] = atom_name_vocab[atom]
            chi_atom14_index_map[i_resi][j_chi][k_atom] = restype_atom14_index_map[i_resi][atom_name_vocab[atom]]
# Masks out non-existent torsions.
chi_masks = chi_atom37_index_map != -1


def get_dihedral(p0, p1, p2, p3):
    """
    Given p0-p3, compute dihedral b/t planes p0p1p2 and p1p2p3.
    """
    assert p0.shape[-1] == p1.shape[-1] == p2.shape[-1] == p3.shape[-1] == 3

    # dx
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    # normals
    n012 = torch.cross(b0, b1)
    n123 = torch.cross(b1, b2)

    # dihedral
    cos_theta = torch.einsum('...i,...i->...', n012, n123) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) + 1e-10)
    sin_theta = torch.einsum('...i,...i->...', torch.cross(n012, n123), b1) / (
            torch.norm(n012, dim=-1) * torch.norm(n123, dim=-1) * torch.norm(b1, dim=-1) + 1e-10)
    theta = torch.atan2(sin_theta, cos_theta)

    return theta


def get_chi_mask(protein, chi_id=None):
    chi_atom14_index = chi_atom14_index_map.to(protein.device)[protein.residue_type]  # (num_residue, 4, 4) 0~13
    chi_atom14_mask = chi_atom14_index != -1
    chi_mask = chi_atom14_mask.all(dim=-1)  # (num_residue, 4)

    chi_atom_position = get_chi_atom_position(protein)  # (num_residue, 4, 4, 3)
    has_atom_mask = ~torch.isnan(chi_atom_position).any(dim=-1).any(dim=-1)  # (num_residue, 4)
    chi_mask = chi_mask & has_atom_mask

    if chi_id is not None:
        chi_mask[:, :chi_id] = False
        chi_mask[:, chi_id + 1:] = False
    return chi_mask


def get_chis(protein, node_position=None):
    if node_position is None:
        node_position = protein.node_position
    chi_atom_position = get_chi_atom_position(protein, node_position)   # (num_residue, 4, 4, 3)
    chis = get_dihedral(*chi_atom_position.unbind(-2))                  # (num_residue, 4)
    chis[~protein.chi_mask] = np.nan
    return chis


def get_chi_atom_position(protein, node_position=None):
    """
    Get atom position for each chi torsion angles of each residue.

    Args:
        protein: Protein object.
        node_position: (num_atom, 3) tensor, atom position.

    Returns:
        chi_atom_position: (num_residue, 4, 4, 3) tensor, atom position for each chi torsion angles of each residue.
        `Nan` indicates that the atom does not exist.
    """
    if node_position is None:
        node_position = protein.node_position
    node_position37 = get_atom37_position(protein, node_position)
    chi_atom37_index = chi_atom37_index_map.to(protein.device)[protein.residue_type]    # (num_residue, 4, 4) 0~36
    chi_atom37_mask = chi_atom37_index == -1
    chi_atom37_index[chi_atom37_mask] = 0
    chi_atom37_index = chi_atom37_index.flatten(-2, -1)                                 # (num_residue, 16)
    chi_atom_position = torch.gather(node_position37, -2,
                                     chi_atom37_index[:, :, None].expand(-1, -1, 3))    # (num_residue, 16, 3)
    chi_atom_position = chi_atom_position.view(-1, 4, 4, 3)                             # (num_residue, 4, 4, 3)
    return chi_atom_position


@torch.no_grad()
def get_chi_periodic_mask(protein, chi_id=None):
    chi_mask = get_chi_mask(protein)
    chi_1pi_periodic_mask = torch.tensor(chi_pi_periodic).to(protein.device)[protein.residue_type]  # [num_residue, 4]
    chi_2pi_periodic_mask = ~chi_1pi_periodic_mask  # [num_residue, 4]
    chi_1pi_periodic_mask = torch.logical_and(chi_mask, chi_1pi_periodic_mask)
    chi_2pi_periodic_mask = torch.logical_and(chi_mask, chi_2pi_periodic_mask)
    for mask in [chi_1pi_periodic_mask, chi_2pi_periodic_mask]:
        if chi_id is not None:
            mask[:, :chi_id] = False
            mask[:, chi_id + 1:] = False
    return chi_1pi_periodic_mask, chi_2pi_periodic_mask


@torch.no_grad()
def set_chis(protein, chis):
    assert chis.shape[0] == protein.num_residue
    assert chis.shape[1] == 4
    cur_chis = get_chis(protein)
    chi_to_rotate = chis - cur_chis
    chi_to_rotate[torch.isnan(chi_to_rotate)] = 0
    rotate_side_chain(protein, chi_to_rotate)
    return protein


@torch.no_grad()
def rotate_side_chain(protein, rotate_angles):
    assert rotate_angles.shape[0] == protein.num_residue
    assert rotate_angles.shape[1] == 4
    node_position14, mask14 = get_atom14_position(protein)  # (num_residue, 14, 3)

    chi_atom14_index = chi_atom14_index_map.to(protein.device)[protein.residue_type]  # (num_residue, 4, 4) 0~13
    chi_atom14_mask = chi_atom14_index != -1
    chi_atom14_index[~chi_atom14_mask] = 0
    for i in range(4):
        atom_1, atom_2, atom_3, atom_4 = chi_atom14_index[:, i, :].unbind(-1)  # (num_residue, )
        atom_2_position = torch.gather(node_position14, -2,
                                       atom_2[:, None, None].expand(-1, -1, 3))  # (num_residue, 1, 3)
        atom_3_position = torch.gather(node_position14, -2,
                                       atom_3[:, None, None].expand(-1, -1, 3))  # (num_residue, 1, 3)
        axis = atom_3_position - atom_2_position
        axis_normalize = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)
        rotate_angle = rotate_angles[:, i, None, None]

        # Rotate all subsequent atoms by the rotation angle
        rotate_atoms_position = node_position14 - atom_2_position  # (num_residue, 14, 3)
        parallel_component = (rotate_atoms_position * axis_normalize).sum(dim=-1, keepdim=True) \
                             * axis_normalize
        perpendicular_component = rotate_atoms_position - parallel_component
        perpendicular_component_norm = perpendicular_component.norm(dim=-1, keepdim=True) + 1e-10
        perpendicular_component_normalize = perpendicular_component / perpendicular_component_norm
        normal_vector = torch.cross(axis_normalize.expand(-1, 14, -1), perpendicular_component_normalize, dim=-1)
        transformed_atoms_position = perpendicular_component * rotate_angle.cos() + \
                                     normal_vector * perpendicular_component_norm * rotate_angle.sin() + \
                                     parallel_component + atom_2_position  # (num_residue, 14, 3)
        assert not transformed_atoms_position.isnan().any()
        chi_mask = chi_atom14_mask[:, i, :].all(dim=-1, keepdim=True)  # (num_residue, 1)
        atom_mask = torch.arange(14, device=protein.device)[None, :] >= atom_4[:, None]  # (num_residue, 14)
        mask = (atom_mask & chi_mask).unsqueeze(-1).expand_as(node_position14)
        node_position14[mask] = transformed_atoms_position[mask]

    protein.node_position[mask14] = node_position14[protein.atom2residue[mask14], protein.atom14index[mask14]]
    return chi_atom14_mask.all(dim=-1)


@torch.no_grad()
def remove_by_chi(protein, chi_id):
    new_protein = protein.clone()
    mask_attrs = ['chi_1pi_periodic_mask', 'chi_2pi_periodic_mask', 'chi_mask']
    for attr in mask_attrs:
        if hasattr(new_protein, attr):
            getattr(new_protein, attr)[:, :chi_id] = 0
            getattr(new_protein, attr)[:, chi_id + 1:] = 0

    if chi_id == 3:
        return new_protein
    else:
        chi_atom14_index = chi_atom14_index_map.to(new_protein.device)[new_protein.residue_type]
        atom_4 = chi_atom14_index[:, chi_id + 1, -1]
        atom_mask = (new_protein.atom14index >= atom_4[new_protein.atom2residue]) & (
                atom_4[new_protein.atom2residue] != -1)
        new_protein = new_protein.subgraph(~atom_mask)
        return new_protein


def get_atom14_position(protein, node_position=None):
    """
    Get the position of 14 atoms for each residue
    Args:
        protein: Protein object
        node_position: (num_atom, 3)

    Returns:
        node_position14: (num_residue, 14, 3)
        mask14: (num_atom,) indicate whether the atom is in the 14 atoms
    """
    if node_position is None:
        node_position = protein.node_position
    atom14index = restype_atom14_index_map.to(protein.device)[
        protein.residue_type[protein.atom2residue], protein.atom_name
    ]  # (num_atom, )
    node_position14 = torch.zeros((protein.num_residue, 14, 3), dtype=torch.float, device=protein.device)
    mask14 = atom14index != -1  # (num_atom, )
    node_position14[protein.atom2residue[mask14], atom14index[mask14], :] = node_position[mask14]
    return node_position14, mask14


def get_atom37_position(protein, node_position=None, return_nan=True):
    """
    Get the position of 37 atoms for each residue

    Args:
        protein: Protein object
        node_position: (num_atom, 3)
        return_nan: whether to return nan if the atom is not in the 37 atoms

    Returns:
        node_position37: (num_residue, 37, 3): nan if the atom is not in the 37 atoms
    """
    if node_position is None:
        node_position = protein.node_position

    if return_nan:
        node_position37 = torch.ones((protein.num_residue, 37, 3), dtype=torch.float, device=protein.device) * np.nan
    else:
        node_position37 = torch.zeros((protein.num_residue, 37, 3), dtype=torch.float, device=protein.device)

    node_position37[protein.atom2residue, protein.atom_name, :] = node_position
    return node_position37


@torch.no_grad()
def set_atom14_position(protein, node_position14):
    """
    Set the position of 14 atoms for each residue

    Args:
        protein: Protein object
        node_position14: (num_residue, 14, 3)

    Returns:
        protein: Protein object
    """
    atom14index = restype_atom14_index_map[
        protein.residue_type[protein.atom2residue], protein.atom_name
    ]  # (num_atom, )
    mask14 = atom14index != -1
    protein.node_position[mask14] = node_position14[protein.atom2residue[mask14], atom14index[mask14]]
    return protein


@torch.no_grad()
def get_rigid_transform(n_xyz, ca_xyz, c_xyz, eps=1e-20):
    """
    Returns a transformation object from reference coordinates.

    Note that this method does not take care of symmetries. If you
    provide the atom positions in the non-standard way, the N atom will
    end up not at [-0.527250, 1.359329, 0.0] but instead at
    [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
    your code.

    Args:
        n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
        ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
        c_xyz: A [*, 3] tensor of carbon xyz coordinates.
    Returns:
        A transformation (rots, translations). After applying the translation and
        rotation to the reference backbone, the coordinates will
        approximately equal to the input coordinates.
    """

    translation = -1 * ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
    norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
    sin_c1 = -c_y / norm
    cos_c1 = c_x / norm
    zeros = sin_c1.new_zeros(sin_c1.shape)
    ones = sin_c1.new_ones(sin_c1.shape)

    c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
    c1_rots[..., 0, 0] = cos_c1
    c1_rots[..., 0, 1] = -1 * sin_c1
    c1_rots[..., 1, 0] = sin_c1
    c1_rots[..., 1, 1] = cos_c1
    c1_rots[..., 2, 2] = 1

    norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
    sin_c2 = c_z / norm
    cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

    c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    c2_rots[..., 0, 0] = cos_c2
    c2_rots[..., 0, 2] = sin_c2
    c2_rots[..., 1, 1] = 1
    c2_rots[..., 2, 0] = -1 * sin_c2
    c2_rots[..., 2, 2] = cos_c2

    c_rots = rot_matmul(c2_rots, c1_rots)
    n_xyz = rot_vec_mul(c_rots, n_xyz)

    _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
    norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
    sin_n = -n_z / norm
    cos_n = n_y / norm

    n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
    n_rots[..., 0, 0] = 1
    n_rots[..., 1, 1] = cos_n
    n_rots[..., 1, 2] = -1 * sin_n
    n_rots[..., 2, 1] = sin_n
    n_rots[..., 2, 2] = cos_n

    rots = rot_matmul(n_rots, c_rots)

    rots = rots.transpose(-1, -2)
    translation = -1 * translation

    return rots, translation


@torch.no_grad()
def init_sidechain(protein, keep_backbone=True):
    """
    Initialize the side chain of the protein from restype_atom14_rigid_group_positions (shape: 21, 14, 3)
    Args:
        protein: Protein object
        keep_backbone: If True, the backbone will not be changed
    """
    node_position14, mask14 = get_atom14_position(protein)  # (num_residue, 14, 3), (num_residue, 14)
    assert mask14[..., 0].all()  # Each Residue should have N atom
    assert mask14[..., 1].all()  # Each Residue should have CA atom
    assert mask14[..., 2].all()  # Each Residue should have C atom
    rots, translation = get_rigid_transform(n_xyz=node_position14[..., 0, :],
                                            ca_xyz=node_position14[..., 1, :],
                                            c_xyz=node_position14[..., 2, :])  # (num_residue, 3, 3), (num_residue, 3)
    # Align the rigid side chain to backbone
    new_node_position14 = rot_vec_mul(rots.unsqueeze(-3), restype_atom14_rigid_group_positions[protein.residue_type]) \
                          + translation.unsqueeze(-2)
    new_node_position14 = new_node_position14 * restype_atom14_mask[protein.residue_type].unsqueeze(-1)

    # Preserve the original position of backbone atoms
    if keep_backbone:
        new_node_position14[..., 0:4] = node_position14[..., 0:4]

    # Set the position of 14 atoms for each residue
    protein = set_atom14_position(protein, new_node_position14)
    return protein


@torch.no_grad()
def randomize(protein):
    torsion_updates = torch.rand((protein.num_residue, 4), device=protein.device) * 2 * np.pi
    rotate_side_chain(protein, torsion_updates)
    return protein


def _rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, mask):
    sd = torch.square(pred_pos_per_residue - true_pos_per_residue).sum(dim=-1)
    msd = sd.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    rmsd = msd.sqrt()
    return rmsd


def _get_symm_atoms(pos_per_residue, residue_type):
    sym_pos_per_residue = pos_per_residue.clone()   # [num_residues, 37, 3]
    for i in range(2):
        atom_to = res_sym_atom_posn.to(residue_type.device)[residue_type][:, i, 0]      # [num_residues]
        atom_from = res_sym_atom_posn.to(residue_type.device)[residue_type][:, i, 1]    # [num_residues]
        sym_pos_per_residue[torch.arange(len(pos_per_residue)), atom_to] = \
            pos_per_residue[torch.arange(len(pos_per_residue)), atom_from]
        sym_pos_per_residue[torch.arange(len(pos_per_residue)), atom_from] = \
            pos_per_residue[torch.arange(len(pos_per_residue)), atom_to]
    return sym_pos_per_residue
