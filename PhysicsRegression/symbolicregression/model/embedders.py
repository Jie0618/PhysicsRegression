from typing import Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from symbolicregression.utils import to_cuda
import torch.nn.functional as F

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def num_encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=env.float_word2id["<PAD>"],
        )
        self.float_scalar_descriptor_len = (2 + self.params.mantissa_len)
        self.total_dimension = self.params.max_input_dimension + self.params.max_output_dimension
        self.float_vector_descriptor_len = self.float_scalar_descriptor_len * self.total_dimension

        self.activation_fn = F.relu
        size = (self.float_vector_descriptor_len + 2)*self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_len

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers: 
            sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings

    def forward(self, sequences: List[Sequence], hints) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self.num_encode(sequences)
        if self.params.use_hints:
            hints = self.hint_encode(hints, self.params.use_hints)
            sequences = [torch.cat((hint, sequence), dim=0) for hint, sequence in zip(hints, sequences)]
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.fc.weight.device.type=="cpu", device=self.env.params.device)
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def num_encode(self, sequences: List[Sequence]) -> List[torch.Tensor]:
        res = []
        for seq in sequences:
            seq_toks = []
            for x, y in seq:
                x_toks = self.env.float_encoder.encode(x)
                y_toks = self.env.float_encoder.encode(y)
                input_dim = int(len(x_toks) / (2 + self.params.mantissa_len))
                output_dim = int(len(y_toks) / (2 + self.params.mantissa_len))
                x_toks = [
                    *x_toks,
                    *[
                        "<INPUT_PAD>"
                        for _ in range(
                            (self.params.max_input_dimension - input_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                y_toks = [
                    *y_toks,
                    *[
                        "<OUTPUT_PAD>"
                        for _ in range(
                            (self.params.max_output_dimension - output_dim)
                            * self.float_scalar_descriptor_len
                        )
                    ],
                ]
                toks = ["<DATA_POINT>", *x_toks, *y_toks, "</DATA_POINT>"]
                seq_toks.append([self.env.float_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        return res

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)
        sent = torch.LongTensor(slen, bs, self.float_vector_descriptor_len+2).fill_(pad_id)
        for i, seq in enumerate(seqs):
            sent[0 : len(seq), i, :] = seq
        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths
    
    def hint_encode(self, hints, use_hints):
        use_hints = use_hints.split(",")
        res = []
        for seq_id in range(len(hints[0])):
            seq_toks = []
            current_hints_idx = 0

            if "units" in use_hints:
                units = hints[current_hints_idx]
                for i, unit in enumerate(units[seq_id]):
                    un = self.env.equation_encoder.units_encode(unit)
                    units_toks = [
                        "<PHYSICAL_UNITS>",
                        f"x_{i}" if i != len(units[seq_id]) - 1 else f"y",
                        *un,
                        "</PHYSICAL_UNITS>",
                        *[
                            "<HINT_PAD>" for _ in range(
                                (self.params.max_input_dimension + self.params.max_output_dimension) 
                                * self.float_scalar_descriptor_len - len(un) - 3 + 2
                            )
                        ],
                    ]
                    seq_toks.append([self.env.float_word2id[tok] for tok in units_toks])
                current_hints_idx += 1

            if "complexity" in use_hints:
                complexity = hints[current_hints_idx]
                for c in complexity[seq_id]:
                    com_toks = [
                        "<COMPLEXITY>", 
                        f"COMPLEXITY:{c}" if c != 0 else "<UNKNOWN_COMPLEXITY>", 
                        "</COMPLEXITY>",
                        *[
                            "<HINT_PAD>" for _ in range(
                                (self.params.max_input_dimension + self.params.max_output_dimension) 
                                * self.float_scalar_descriptor_len - 3 + 2
                            )
                        ],
                    ]
                    seq_toks.append([self.env.float_word2id[tok] for tok in com_toks])
                current_hints_idx += 1

            if "unarys" in use_hints:
                unarys = hints[current_hints_idx]
                unary_toks = [
                    "<UNARY>",
                    *unarys[seq_id],
                    "</UNARY>",
                    *[
                        "<HINT_PAD>" for _ in range(
                            (self.params.max_input_dimension + self.params.max_output_dimension) 
                            * self.float_scalar_descriptor_len - len(unarys[seq_id]) - 2 + 2
                        )
                    ],
                ]
                seq_toks.append([self.env.float_word2id[tok] for tok in unary_toks])
                current_hints_idx += 1

            if "add_structure" in use_hints:
                add_structure = hints[current_hints_idx]
                for a in add_structure[seq_id]:
                    add_tokes = [
                        f"x_{j}" for j in a 
                    ]
                    add_struc_toks = [
                        "<ADD_STRUCTURE>",
                        *add_tokes,
                        "</ADD_STRUCTURE>",
                        *[
                            "<HINT_PAD>" for _ in range(
                                (self.params.max_input_dimension + self.params.max_output_dimension) 
                                * self.float_scalar_descriptor_len - len(add_tokes) - 2 + 2
                            )
                        ],
                    ]
                    seq_toks.append([self.env.float_word2id[tok] for tok in add_struc_toks])
                current_hints_idx += 1

            if "mul_structure" in use_hints:
                mul_structure = hints[current_hints_idx]
                for m in mul_structure[seq_id]:
                    mul_tokes = [
                        f"x_{j}" for j in m 
                    ]
                    mul_struc_toks = [
                        "<MUL_STRUCTURE>",
                        *mul_tokes,
                        "</MUL_STRUCTURE>",
                        *[
                            "<HINT_PAD>" for _ in range(
                                (self.params.max_input_dimension + self.params.max_output_dimension) 
                                * self.float_scalar_descriptor_len - len(mul_tokes) - 2 + 2
                            )
                        ],
                    ]
                    seq_toks.append([self.env.float_word2id[tok] for tok in mul_struc_toks])
                current_hints_idx += 1
            
            if "consts" in use_hints:
                consts = hints[current_hints_idx]
                for _value, _units in consts[seq_id]:
                    const_toks = [
                        *self.env.float_encoder.encode(np.array([_value])),
                        *self.env.equation_encoder.units_encode(_units),
                    ]  if _value != "pi" else [
                        "pi",
                        *self.env.equation_encoder.units_encode(_units),
                    ]
                    used_const_toks = [
                        "<USED_CONST>",
                        *const_toks,
                        "</USED_CONST>",
                        *[
                            "<HINT_PAD>" for _ in range(
                                (self.params.max_input_dimension + self.params.max_output_dimension) 
                                * self.float_scalar_descriptor_len - len(const_toks) - 2 + 2
                            )
                        ],
                    ]
                    seq_toks.append([self.env.float_word2id[tok] for tok in used_const_toks])
                current_hints_idx += 1

            res.append(torch.LongTensor(seq_toks))
        return res
    
