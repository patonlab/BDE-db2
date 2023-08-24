from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from nfp.preprocessing.features import get_ring_size
from tqdm.auto import tqdm

import rdkit.Chem  # noqa: F401 isort:skip
import nfp  # isort:skip


tqdm.pandas()

def atom_featurizer(atom):
    """ Return an integer hash representing the atom type
    """

    return str((
        atom.GetSymbol(),
        atom.GetNumRadicalElectrons(),
        atom.GetFormalCharge(),
        atom.GetChiralTag(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()

preprocessor = nfp.SmilesBondIndexPreprocessor(
    atom_features=atom_featurizer, bond_features=bond_featurizer)
    

if __name__ == '__main__':
    
    data_dir = "/home/svss/projects/Project-BDE/20220221-new-models/"
    bde = pd.read_csv('20210217_rdf_with_multi_halo_cfc.csv.gz')

    def create_example(df, train=True):

        smiles = df.molecule.iloc[0]
        input_dict = preprocessor(smiles, train=train)

        targets = (
            df.set_index("bond_index")[["bde", "bdfe"]]
            .reindex(np.arange(input_dict["bond_indices"].max() + 1))
            .values
        )

        input_dict["output"] = targets

        return input_dict

    mol_df = pd.DataFrame(bde.groupby("molecule").set.first())

    train = (
        bde[bde.set == "train"]
        .groupby("molecule")
        .progress_apply(partial(create_example, train=True))
    )

    valid = (
        bde[bde.set != "train"]
        .groupby("molecule")
        .progress_apply(partial(create_example, train=False))
    )

    inputs = train.append(valid)
    inputs.name = "model_inputs"
    mol_df = mol_df.join(inputs)

    mol_df.to_pickle(Path(data_dir, "20220221_tfrecords_multi_halo_cfc/model_inputs.p"))
    preprocessor.to_json(Path(data_dir, "20220221_tfrecords_multi_halo_cfc/preprocessor.json"))
