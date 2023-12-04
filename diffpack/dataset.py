import glob
import logging
import os

import torch
from rdkit import Chem
from torchdrug import data
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from tqdm import tqdm

from diffpack import rotamer
from diffpack.rotamer import get_chi_mask, atom_name_vocab, bb_atom_name

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


@R.register("datasets.SideChainDataset")
class SideChainDataset(data.ProteinDataset):
    processed_file = None
    exclude_pdb_files = []

    def __init__(self, path=None, pdb_files=None, verbose=1, **kwargs):
        if path is not None:
            logger.info("Loading dataset from folder %s" % path)
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.path = path
            pkl_file = os.path.join(path, self.processed_file)

            if os.path.exists(pkl_file):
                logger.info("Found existing pickle file %s" % pkl_file
                            + ". Loading from pickle file (this may take a while)")
                self.load_pickle(pkl_file, verbose=verbose, **kwargs)
            else:
                logger.info("No pickle file found. Loading from pdb files (this may take a while)"
                            + " and save to pickle file %s" % pkl_file)
                pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
                self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
                self.save_pickle(pkl_file, verbose=verbose)
        elif pdb_files is not None:
            logger.info("Loading dataset from pdb files")
            pdb_files = [os.path.expanduser(pdb_file) for pdb_file in pdb_files]
            pdb_files = [pdb_file for pdb_file in pdb_files if pdb_file.endswith(".pdb")]
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)

        # Filter out proteins with no residues
        indexes = [i for i, (protein, pdb_file) in enumerate(zip(self.data, self.pdb_files))
                   if (protein.num_residue > 0).all() and os.path.basename(pdb_file) not in self.exclude_pdb_files]
        self.data = [self.data[i] for i in indexes]
        self.sequences = [self.sequences[i] for i in indexes]
        self.pdb_files = [self.pdb_files[i] for i in indexes]

    def load_pdbs(self, pdb_files, transform=None, lazy=False, verbose=0, sanitize=True, removeHs=True, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file, sanitize=sanitize, removeHs=removeHs)
                if not mol:
                    logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
                protein = data.Protein.from_molecule(mol, **kwargs)
                if not protein:
                    logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file)
                    continue
            else:
                protein = None
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)

    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], **self.kwargs)
        else:
            protein = self.data[index].clone()
        protein = protein.subgraph(protein.atom_name < 37)

        with protein.atom():
            # Init atom14 index map
            protein.atom14index = rotamer.restype_atom14_index_map[
                protein.residue_type[protein.atom2residue], protein.atom_name
            ]  # [num_atom, 14]

        with protein.residue():
            # Init residue features
            protein.residue_feature = functional.one_hot(protein.residue_type, 21)  # [num_residue, 21]

            # Init residue masks
            chi_mask = get_chi_mask(protein)
            chi_1pi_periodic_mask = torch.tensor(rotamer.chi_pi_periodic)[protein.residue_type]
            chi_2pi_periodic_mask = ~chi_1pi_periodic_mask
            protein.chi_mask = chi_mask
            protein.chi_1pi_periodic_mask = torch.logical_and(chi_mask, chi_1pi_periodic_mask)  # [num_residue, 4]
            protein.chi_2pi_periodic_mask = torch.logical_and(chi_mask, chi_2pi_periodic_mask)  # [num_residue, 4]

            # Init atom37 features
            protein.atom37_mask = torch.zeros(protein.num_residue, len(atom_name_vocab), device=protein.device,
                                              dtype=torch.bool)  # [num_residue, 37]
            protein.atom37_mask[protein.atom2residue, protein.atom_name] = True
            protein.sidechain37_mask = protein.atom37_mask.clone()  # [num_residue, 37]
            protein.sidechain37_mask[:, bb_atom_name] = False
        item = {"graph": protein}

        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def from_pdb_files(pdb_files, verbose=1, **kwargs):
        return SideChainDataset(pdb_files, verbose=verbose, **kwargs)

    def __repr__(self):
        lines = ["#sample: %d" % len(self)]
        return "%s(  %s)" % (self.__class__.__name__, "\n  ".join(lines))


@R.register("datasets.BC40Train")
class BC40Train(SideChainDataset):
    processed_file = "bc40_train.pkl.gz"


@R.register("datasets.BC40Valid")
class BC40Valid(SideChainDataset):
    processed_file = "bc40_valid.pkl.gz"


@R.register("datasets.CASP13")
class CASP13(SideChainDataset):
    exclude_pdb_files = ['T0999.pdb']
    processed_file = "casp13.pkl.gz"


@R.register("datasets.CASP14")
class CASP14(SideChainDataset):
    exclude_pdb_files = ['T1029.pdb', 'T1041.pdb', 'T1044.pdb', 'T1050.pdb',
                         'T1061.pdb', 'T1076.pdb', 'T1101.pdb', ]
    processed_file = "casp14.pkl.gz"