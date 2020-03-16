from __future__ import annotations
from typing import Any, Iterable, Optional

import abc
import os
import numpy as np
import torch

__all__ = [
    "Argmax",
    "Compose",
    "DigraphToTensors",
    "Expand",
    "Identity",
    "Lookup",
    "NormalizePerTensor",
    "PowerSeries",
    "Transpose",
    "ZeroMeanPerTensor",
]


class Transform:
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "The function __init__ must be overwritten for this class."
        )

    @abc.abstractmethod
    def __call__(self, x):
        "Transform which is applied to an input tensor."


class Argmax(Transform):
    def __init__(
        self, dim: Optional[int] = None, keepdim: Optional[bool] = False
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, x: torch.Tensor) -> torch.LongTensor:
        return x.argmax(dim=self.dim, keepdim=self.keepdim)


class Compose(Transform):
    def __init__(self, *transforms: Iterable[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class DigraphToTensors(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: networkx.DiGraph):
        nodes = torch.Tensor(np.vstack([x.nodes[n]["x"] for n in x.nodes]))
        edge_index = torch.Tensor(list(x.edges)).long().t().contiguous()

        return nodes, edge_index


class Expand(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


class Identity(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Lookup(Transform):
    def __init__(self, lookup_dict) -> None:
        self.lookup_dict = lookup_dict

    def __call__(self, x) -> Any:
        return self.lookup_dict[x]


class NormalizePerTensor(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x -= torch.mean(x)
        x /= torch.std(x)
        return x


class PowerSeries(Transform):
    def __init__(self, degree: int, dim: Optional[int] = -1) -> None:
        self.degree = degree
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tensors = tuple(x ** k for k in range(1,self.degree + 1))
        try:
            return torch.cat(tensors=tensors, dim=self.dim)
        except __:
            return torch.stack(tensors=tensors,dim=self.dim)


class Transpose(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.t()


class ZeroMeanPerTensor(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x -= torch.mean(x)
        return x


# --------------------------------------------------------- #

# import rdkit, rdkit.Chem

# from .transform import Transform


# class PDBToRDKMol(Transform):
#     def __init__(self):
#         pass

#     def __call__(self, x):
#         return rdkit.Chem.MolFromPDBFile(molFileName=x)

# ------------------------------------------------------------ #

# import numpy as np

# import torch

# import networkx as nx


# class RDKMolToDigraph(Transform):
#     def __init__(self, **kwargs):
#         # FIXME: revise whole class
#         pass

#     def one_hot(self, Z, alphabet):
#         # FIXME: alphabet must be tuple

#         t = torch.zeros(size=(len(alphabet),)).long()

#         t[alphabet.index(Z)] = 1

#         return t

#     def __call__(self, x):
#         # FIXME: this doesn't work for any molecule with just one heavy atom, e.g. methane (fails at nx.center due to nx.eccentricity returning {} due to lack of bonds)
#         # FIXME: rewrite - more efficient version elsewhere...
#         # atom_index_list = [atom.GetIdx() for atom in x.GetAtoms()]

#         bonds = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in x.GetBonds())
#         g = nx.Graph()
#         g.add_edges_from(bonds)

#         alphabet = (
#             1,
#             6,
#             7,
#             8,
#             9,
#         )  # FIXME: this is a really bad hardcode for QM9 #tuple(set(Z for Z,_,_ in atomic_properties.values()))
#         hydrogen_nodes = (atom_index for atom_index, Z in enumerate(alphabet) if Z == 1)
#         g.remove_nodes_from(hydrogen_nodes)
#         remap = {n: i for i, n in enumerate(g.nodes)}
#         g = nx.relabel_nodes(G=g, mapping=remap, copy=True)
#         # atomic properties: Z, formal charge, number of bound hydrogens
#         atomic_properties = {
#             atom.GetIdx(): (
#                 atom.GetAtomicNum(),
#                 atom.GetFormalCharge(),
#                 atom.GetTotalNumHs(includeNeighbors=True),
#             )
#             for atom in x.GetAtoms()
#         }
#         for atom_index, (Z, formal_charge, num_hydrogens) in atomic_properties.items():
#             atomic_properties[atom_index] = list(
#                 self.one_hot(Z=Z, alphabet=alphabet).numpy()
#             ) + [formal_charge, num_hydrogens]
#         atomic_properties = {
#             remap[k]: v for k, v in atomic_properties.items() if k in remap
#         }

#         central_vertex = np.random.choice(
#             nx.center(G=g)
#         )  # find the center of the graph to minimize propagation length

#         dg = nx.DiGraph()
#         directed_edges = (
#             (p, v)
#             for v, p_list in nx.predecessor(G=g, source=central_vertex).items()
#             for p in p_list
#         )

#         dg.add_edges_from(directed_edges)
#         nx.set_node_attributes(dg, values=atomic_properties, name="x")

#         return dg

# ------------------------------------------------------------------------- #

# class NormalizePerDataset(Transform):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.mean = self._compute_mean()

#     def _compute_mean(self):
#         M = 0
#         S = 0
#         num = 0
#         for name in os.listdir(self.data_dir):
#             path = os.path.join(self.data_dir, name)
#             if os.path.isfile(path):
#                 x = torch.load(path)
#                 num += 1
#                 Mnew = M + (x - M) / num
#                 S += (x - M) * (x - Mnew)
#                 M = Mnew

#         self.mean = M
#         self.std = torch.sqrt(S / (num - 1))

#     def __call__(self, x):
#         x -= self.mean
#         x /= self.std
#         return x

# ----------------------------------------------------------- #

# class ZeroMeanPerDataset:
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.mean = self._compute_mean()

#     def _compute_mean(self):
#         sum = 0
#         num = 0
#         for name in os.listdir(self.data_dir):
#             path = os.path.join(self.data_dir, name)
#             if os.path.isfile(path):
#                 sum += torch.load(path)
#                 num += 1

#         self.mean = sum / num

#     def __call__(self, x):
#         x -= self.mean
#         return x

# ----------------------------------------------------------- #

# import os
# import string
# import unicodedata
#
# import torch
#
# from .transform import Transform
#
# class UnicodeToASCIIOneHot(Transform):
#     def __init__(self, data_dir, **kwargs):
#         self.all_letters = string.ascii_letters + " .,;'"
#         self.data_dir = data_dir
#         self.alphabet = self._make_alphabet()
#
#     def _make_alphabet(self):
#         alphabet = set()
#
#         for name in os.listdir(self.data_dir):
#             path = os.path.join(self.data_dir,name)
#             if os.path.isfile(path):
#                 with open(path,'r') as f:
#                     alphabet.update(list(self._unicode_to_ascii(f.read())))
#
#         return list(alphabet)
#
#     def _unicode_to_ascii(self, x):
#         return ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn' and c in self.all_letters)
#
#     def __call__(self, x):
#         tensor = torch.zeros(len(x), len(self.all_letters))
#
#         for index,letter in enumerate(self._unicode_to_ascii(x)):
#             tensor[index][self.alphabet.index(letter)] = 1
#
#         return tensor
#
# class UnicodeToASCII(Transform):
#     def __init__(self, **kwargs):
#         self.all_letters = string.ascii_letters + " .,;'"
#
#     def __call__(self, x):
#         return ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn' and c in self.all_letters)
#
# class OneHot(Transform):
#     def __init__(self, data_dir, **kwargs):
#         self.data_dir = data_dir
#         self.alphabet = self._make_alphabet()
#
#     def _make_alphabet(self):
#         alphabet = set()
#
#         for name in os.listdir(self.data_dir):
#             path = os.path.join(self.data_dir,name)
#             if os.path.isfile(path):
#                 with open(path,'r') as f:
#                     alphabet.update(list(f.read()))
#
#         return tuple(alphabet)
#
#     def __call__(self, x):
#         tensor = torch.zeros(len(x), 1, len(self.alphabet))
#
#         for index,letter in enumerate(x):
#             tensor[index][0][self.alphabet.index(letter)] = 1
#
#         return tensor
#
# class OneHotTensor(Transform):
#     def __init__(self, data_dir, **kwargs):
#         # FIXME: rework this class to apply equally to strings and standard tensors, then merge with OneHot
#         self.data_dir = data_dir
#         self.alphabet = self._make_alphabet()
#         print(self.alphabet)
#
#     def _make_alphabet(self):
#         alphabet = set()
#
#         for name in os.listdir(self.data_dir):
#             path = os.path.join(self.data_dir,name)
#             if os.path.isfile(path):
#                 t = torch.load(path)
#                 alphabet.update((x.item() for x in t) if t.dim() > 0 else (t.item(),))
#
#         return tuple(alphabet)
#
#     def __call__(self, x):
#         tensor = torch.zeros(len(x), 1, len(self.alphabet))
#
#         for index,letter in enumerate(x):
#             tensor[index][0][self.alphabet.index(letter)] = 1
#
#         return tensor
#
# # class FourierSeries(FeatureFunction):
# #     def __init__(self, real=False):
# #         self.fft_func = np.fft.rfft if real else np.fft.fft
# #         self.ifft_func = np.fft.irfft if real else np.fft.ifft
# #
# #     def featurize(self, x):
# #         ck = self.fft_func(x)
# #
# # class FourierSeriesTruncate(FourierSeries):
# #     def __init__(self, n_trunc, real=False, truncate_mode='smallest'):
# #         super(FourierSeriesTruncate, self).__init__(real=real)
# #
# #         truncate_funcs = {'smallest': self.truncate_smallest, 'high_freq': self.truncate_high_freq, 'low_freq': self.truncate_low_freq}
# #
# #         if truncate_mode not in truncate_funcs.keys():
# #             raise ValueError('Argument truncate_mode must be one of the following: {}.'.format(', '.join(truncate_modes)))
# #         if not isinstance(n_trunc, int) or n_trunc < 0:
# #             raise ValueError('Argument n_trunc must be a nonnegative integer.')
# #
# #         self.n_trunc = n_trunc
# #         self.truncate = truncate_funcs[truncate_mode]
# #
# #     def featurize(self, x):
# #         return self.truncate(self.fft_func(x))
# #
# #     def truncate_smallest(self, ck):
# #         truncated = ck
# #
# #         if self.n_trunc == 0:
# #             truncate_indices = []
# #         else:
# #             truncate_indices = ck.argsort()[-self.n_trunc:]
# #
# #         truncated[truncate_indices] = 0
# #
# #         return truncated
# #
# #     def truncate_high_freq(self, ck):
# #         truncated = ck
# #
# #         if self.n_trunc > 0:
# #             truncated[-self.n_trunc:] = 0
# #
# #         return truncated
# #
# #     def truncate_low_freq(self, ck):
# #         truncated = ck
# #
# #         if self.n_trunc > 0:
# #             truncated[:self.n_trunc] = 0
# #
# #         return truncated
# #
# # class Digital(FeatureFunction):
# #     def __init__(self, filepath, data_key, num_intervals, num_bins):
# #         assert False, 'TODO: Fix how intervals work/are computed in Digital.featurize'
# #         self.num_intervals = num_intervals
# #         self.num_bins = num_bins
# #
# #     def bin_data(self, x):
# #         data_array = np.array(x)
# #         truncated_array = data_array[:self.num_bins*(data_array.size//self.num_bins)]
# #
# #         return truncated_array.reshape(self.num_bins,-1).mean(axis=1)
# #
# #     def featurize(self, x):
# #         binned_data = self.bin_data(x)
# #         thresh_vals = [max(x)/3, 2*max(x)/3]
# #
# #         return np.digitize(x=binned_data, bins=thresh_vals)
# #
# # # class OneHot(FeatureFunction):
# # #     def __init__(self, filepath, x_key, alphabet_pattern_str):
# # #         self.alphabet_pattern = re.compile(alphabet_pattern_str)
# # #         self.alphabet, self.pad_to_length = self.make_alphabet(filepath=filepath, x_key=x_key)
# # #
# # #     def make_alphabet(self, filepath, x_key):
# # #         alphabet = set()
# # #
# # #         max_x_length = 0
# # #
# # #         file = h5py.File(filepath, 'r')
# # #         for x in file.get(x_key).value[:,0]:
# # #             max_x_length = len(x) if len(x) > max_x_length else max_x_length
# # #             alphabet.update(self.alphabet_pattern.split(x))
# # #         file.close()
# # #
# # #         return list(alphabet), max_x_length
# # #
# # #     def featurize(self, x):
# # #         one_hot_rep = np.zeros(shape=(self.pad_to_length,len(self.alphabet)))
# # #
# # #         for ind, c in enumerate(x):
# # #             one_hot_rep[ind,:] = [1 if c == alpha else 0 for alpha in self.alphabet]
# # #
# # #         return one_hot_rep
# #
# # class SMILESOneHot(OneHot):
# #     def __init__(self, filepath, x_key):
# #         super(SMILESOneHot, self).__init__(filepath=filepath, x_key=x_key, alphabet_pattern_str='(\[[^[]+\]|[^[])')
# #
# # # class JCAMPProps(FeatureFunction):
# # #     def __init__(self, filepath):
# # #         super(JCAMPProps, self).__init__(filepath=filepath)
# # #
# # #     def featurize(self, x):
# # #         feature_vectors = []
# # #         props = jct.get_properties(filepath)
# # #         feature_vectors.extend(props['moleuclar_weight'])
# # #         return feature_vectors
# # # class JCAMPStatisticalMoments(FeatureFunction):
# # #     def __init__(self, num_moments):
# # #         self.num_moments = num_moments
# # #
# # #     def label(self, filepath):
# # #         raw_data = jct.get_raw_data(filepath)
# # #         wavs, spec = jct.clean_raw_data(raw_data)
# # #         return [spt.compute_moment(moment=i+1, y_dat=spec, x_dat=wavs) for i in range(self.num_moments)]
# # #
# # # class JCAMPIntegrals(FeatureFunction):
# # #     def __init__(self):
# # #         pass
# # #
# # #     def label(self, filepath):
# # #         return spt.ir_vis_uv(filepath=filepath)
