from distutils.log import INFO
from logging import getLogger
import os
import io
import sys
import copy
import json
from typing import Optional, List, Dict
from collections import deque, defaultdict
import time
import traceback
import pickle

# import math
import numpy as np
import symbolicregression.envs.encoders as encoders
import symbolicregression.envs.generators as generators
from symbolicregression.envs.generators import all_operators
import symbolicregression.envs.simplifiers as simplifiers
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import collections
from .utils import *
from ..utils import bool_flag, timeout, MyTimeoutError
import math
import scipy

SPECIAL_WORDS = [
    "<EOS>",
    "<SEP>",
    "<X>",
    "</X>",
    "<Y>",
    "</Y>",
    "</POINTS>",
    "<INPUT_PAD>",
    "<OUTPUT_PAD>",
    "<PAD>",
    "(",
    ")",
    "SPECIAL",
    "OOD_unary_op",
    "OOD_binary_op",
    "OOD_constant",

    #special word for hints
    "<HINT_PAD>",
    "<PHYSICAL_UNITS>", "</PHYSICAL_UNITS>",
    "<COMPLEXITY>", "</COMPLEXITY>",
    "<UNARY>", "</UNARY>",
    "<ADD_STRUCTURE>", "</ADD_STRUCTURE>",
    "<MUL_STRUCTURE>", "</MUL_STRUCTURE>",
    "<USED_CONST>", "</USED_CONST>",

    "<UNKNOWN_PHYSICAL_UNITS>",
    "<UNKNOWN_COMPLEXITY>",

    "<DATA_POINT>", "</DATA_POINT>",
]
logger = getLogger()

SKIP_ITEM = "SKIP_ITEM"


class FunctionEnvironment(object):

    TRAINING_TASKS = {"functions"}

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_size = None
        self.float_tolerance = 10 ** (-params.float_precision)
        self.additional_tolerance = [
            10 ** (-i) for i in range(params.float_precision + 1)
        ]
        assert (
            params.float_precision + 1
        ) % params.mantissa_len == 0, "Bad precision/mantissa len ratio"

        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words
        
        #TODO:
        #i think we can preserve just one encoder
        #so here we just redirect float_words to equation_words
        self.float_words = self.equation_words

        self.simplifier = simplifiers.Simplifier(self.generator)

        # number of words / indices
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}

        for ood_unary_op in self.generator.extra_unary_operators:
            self.equation_word2id[ood_unary_op] = self.equation_word2id["OOD_unary_op"]
        for ood_binary_op in self.generator.extra_binary_operators:
            self.equation_word2id[ood_binary_op] = self.equation_word2id[
                "OOD_binary_op"
            ]
        if self.generator.extra_constants is not None:
            for c in self.generator.extra_constants:
                self.equation_word2id[c] = self.equation_word2id["OOD_constant"]

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(
            f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words"
        )

    def mask_from_seperator(self, x, sep):
        sep_id = self.float_word2id[sep]
        alen = (
            torch.arange(x.shape[0], dtype=torch.long, device=x.device)
            .unsqueeze(-1)
            .repeat(1, x.shape[1])
        )
        sep_id_occurence = torch.tensor(
            [
                torch.where(x[:, i] == sep_id)[0][0].item()
                if len(torch.where(x[:, i] == sep_id)[0]) > 0
                else -1
                for i in range(x.shape[1])
            ]
        )
        mask = alen > sep_id_occurence
        return mask

    def batch_equations(self, equations, units=None, decode_physical_units=None):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        assert decode_physical_units in [None, "single-seq", "double-seq"]

        lengths = torch.LongTensor([2 + len(eq) for eq in equations])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.float_word2id["<PAD>"]
        )
        sent[0] = self.equation_word2id["<EOS>"]
        for i, eq in enumerate(equations):
            sent[1 : lengths[i] - 1, i].copy_(eq)
            sent[lengths[i] - 1, i] = self.equation_word2id["<EOS>"]
        
        if decode_physical_units == "double-seq": 
            sent2 = torch.LongTensor(lengths.max().item(), lengths.size(0), 5).fill_(
                self.float_word2id["<PAD>"]
            )
            sent2[0] = self.equation_word2id["<EOS>"]
            for i, eq in enumerate(units):
                sent2[1 : lengths[i] - 1, i, :].copy_(eq)
                sent2[lengths[i] - 1, i, :] = self.equation_word2id["<EOS>"]
            return sent, lengths, sent2
        else:
            return sent, lengths, None

    def word_to_idx(self, words, float_input=True, unit_input=False):
        if unit_input:
            return [
                torch.LongTensor([
                    [self.float_word2id[dim] for dim in point]
                    for point in seq
                ])
                for seq in words
            ]
        if float_input:
            return [
                [
                    torch.LongTensor([self.float_word2id[dim] for dim in point])
                    for point in seq
                ]
                for seq in words
            ]
        else:
            return [
                torch.LongTensor([self.equation_word2id[w] for w in eq]) for eq in words
            ]

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def idx_to_infix(self, lst, is_float=True, str_array=True):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            idx_to_words = [self.equation_id2word[int(term)] for term in lst]
        return self.word_to_infix(idx_to_words, is_float, str_array)

    def gen_expr(
        self,
        train,
        input_length_modulo=-1,
        nb_binary_ops=None,
        nb_unary_ops=None,
        input_dimension=None,
        output_dimension=None,
        n_input_points=None,
        input_distribution_type=None,
        datatype="train"
    ):
        errors = defaultdict(int)
        if not train or self.params.use_controller:
            if nb_unary_ops is None:
                nb_unary_ops = self.rng.randint(
                    self.params.min_unary_ops, self.params.max_unary_ops + 1
                )
            if input_dimension is None:
                input_dimension = self.rng.randint(
                    self.params.min_input_dimension, self.params.max_input_dimension + 1
                )
        while True:
            try:
                expr, error = self._gen_expr(
                    train,
                    input_length_modulo=input_length_modulo,
                    nb_binary_ops=nb_binary_ops,
                    nb_unary_ops=nb_unary_ops,
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    n_input_points=n_input_points,
                    input_distribution_type=input_distribution_type,
                    datatype=datatype
                )
                if error:
                    errors[error[0]] += 1
                    assert False
                return expr, errors
            except:
                if self.params.debug:
                    # print(expr['tree'])
                    # print(traceback.format_exc())
                    pass
                    # print(error)
                # self.errors["gen expr error"]+=1
                message = traceback.format_exc()
                #print(message)
                continue

    #@timeout(1)
    def _gen_expr(
        self,
        train,
        input_length_modulo=-1,
        nb_binary_ops=None,
        nb_unary_ops=None,
        input_dimension=None,
        output_dimension=None,
        n_input_points=None,
        input_distribution_type=None,
        datatype="train"
    ):
        
        params = self.params
        
        # -------------------------------- GENERATE EXPRESSION TREE --------------------------------
        (
            original_expr,
            tree,
            original_input_dimension,
            output_dimension,
            unary_ops,
            binary_ops,
            real_variables,
            expr_idx,
            variable_consts, 
            variable_consts_units,
        ) = self.generator.generate_multi_dimensional_tree(
            rng=self.rng,
            expr=params.expr_data,
            datatype=datatype,
        )

        # -------------------------------- CHECK TREE VALIDITY --------------------------------
        # is this necessary?

        nb_unary_ops = [len(u) for u in unary_ops]
        nb_binary_ops = [len(b) for b in binary_ops]

        if tree is None:
            return {"tree": tree}, ["bad tree"]
        sum_binary_ops = max(nb_binary_ops)
        sum_unary_ops = max(nb_unary_ops)
        sum_ops = sum_binary_ops + sum_unary_ops
        #input_dimension = self.generator.relabel_variables(tree)
        #if input_dimension == 0 or (
        #    params.enforce_dim and original_input_dimension > input_dimension
        #):
        input_dimension = original_input_dimension
        if input_dimension == 0:
            return {"tree": tree}, ["bad input dimension"]

        for op in params.operators_to_not_repeat.split(","):
            if op and tree.prefix().count(op) > 1:
                return {"tree": tree}, ["ops repeated"]

        if len(real_variables) > 10:
            return {"tree": tree}, ["too many variables"]

        in_out_dimensions = {
            "input_dimension": input_dimension,
            "output_dimension": output_dimension,
        }

        # -------------------------------- SAMPLE XY DATAPOINTS --------------------------------
        if n_input_points is None:
            n_input_points = (
                params.max_len
                if not train
                else self.rng.randint(
                    params.max_len // 2,
                    params.max_len + 1,
                )
            )

        if train:
            n_prediction_points = 0
        else:
            n_prediction_points = params.n_prediction_points

        input_distribution_type_to_int = {"gaussian": 0, "uniform": 1}
        if input_distribution_type is None:
            input_distribution_type = (
                "gaussian" if self.rng.random() < 0.5 else "uniform"
            )
        n_centroids = self.rng.randint(1, self.params.max_centroids)

        if params.prediction_sigmas is None:
            prediction_sigmas = []
        else:
            prediction_sigmas = [
                float(sigma) for sigma in params.prediction_sigmas.split(",")
            ]

        tree, datapoints = self.generator.generate_datapoints(
            tree=tree,
            rng=self.rng,
            input_dimension=in_out_dimensions["input_dimension"],
            n_input_points=n_input_points,
            n_prediction_points=n_prediction_points,
            prediction_sigmas=prediction_sigmas,
            input_distribution_type=input_distribution_type,
            n_centroids=n_centroids,
            max_trials=params.max_trials,
        )

        if datapoints is None:
            #print(original_expr)
            return {
                "tree": tree,
                "original expr": original_expr,
                "expr_idx": expr_idx
            }, ["generation error"]

        x_to_fit, y_to_fit = datapoints["fit"]
        predict_datapoints = copy.deepcopy(datapoints)
        del predict_datapoints["fit"]

        all_outputs = np.concatenate([y for k, (x, y) in datapoints.items()])

        ##output noise added to y_to_fit
        try:
            gamma = (
                self.rng.uniform(0, self.params.train_noise_gamma)
                if train
                else self.params.eval_noise_gamma
            )
            if self.params.eval_noise_type == "multiplicative":
                norm = scipy.linalg.norm(
                    (np.abs(all_outputs) + 1e-100) / np.sqrt(all_outputs.shape[0])
                )
                noise = gamma * norm * self.rng.randn(*y_to_fit.shape)
            elif self.params.eval_noise_type == "additive":
                noise = gamma * y_to_fit * self.rng.randn(*y_to_fit.shape)
            y_to_fit += noise
        except Exception as e:
            print(e, "norm computation error")
            return {"tree": tree}, ["norm computation error"]
        

        # -------------------------------- SAMPLE HINTS --------------------------------                
        use_hints = params.use_hints.split(",")

        #physical units
        if "units" in use_hints:
            if params.sample_dimension_hints == None:
            
                zero_units_prob = 0.06
                if self.rng.random() < zero_units_prob:
                    zero_units = True
                    units = [np.array([0,0,0,0,0]) for k in range(len(real_variables))]
                    y_unit = np.array([0, 0, 0, 0, 0])
                else:
                    zero_units = False
                    units = [
                        self.generator.physical_units_dic[k] 
                        for k in real_variables
                    ]
                    y_unit = tree.unit

                unknown_units_prob = 0.0
                if self.rng.random() < unknown_units_prob:
                    units += ["<UNKNOWN_PHYSICAL_UNITS>"]
                else:
                    units += [y_unit]
            
            else:

                units = params.sample_dimension_hints
                units = [d.split(",") for d in units.split(";")]

                assert len(units) == len(real_variables) + 1, "wrong physical units num for variables"
        
        else:
            zero_units = False
            units = None

        #complexity
        if "complexity" in use_hints:
            if params.sample_complexity_hints == None:
            
                sigma_complexity_sample = 3
                unknown_complexity_prob = 0.1

                sigma_complexity = self.generator.compute_complexity2(tree) + self.rng.normal(0, sigma_complexity_sample)
                if sigma_complexity <= 12:
                    type_complexity = "simple"
                elif sigma_complexity <= 18:
                    type_complexity = "middle"
                else:
                    type_complexity = "hard"
                complexity = [type_complexity if self.rng.random() > unknown_complexity_prob else 0]
            
            else:

                assert params.sample_complexity_hints in ["simple", "middle", "hard", "unknown"]

                complexity = [params.sample_complexity_hints if params.sample_complexity_hints != "unknown" else 0]     
        
        else:
            complexity = None

        #unarys
        if "unarys" in use_hints:
            if params.sample_unary_hints == None:
            
                unary_abandon_prob = 0.25

                unary_ops = [
                        u for u in set(unary_ops[0]) 
                        if self.rng.random() > unary_abandon_prob 
                        and u not in ["neg", "inv", "sqrt", "pow2", "pow3", "pow4", "pow5"]
                    ]

                unary_ops = [
                    u for u in self.generator.unary if u in unary_ops
                ]

            else:

                unary_ops = params.sample_unary_hints
                if unary_ops == "": unary_ops = []
                else:unary_ops = unary_ops.split(",")

                assert isinstance(unary_ops, list) and \
                    all([u in self.generator.unary for u in unary_ops]) and\
                    all([u not in ["neg", "inv"] for u in unary_ops])
        
        else:
            unary_ops = None

        #add/mul structure
        if "add_structure" in use_hints or "mul_structure" in use_hints:
            if params.sample_structure_hints == None:
            
                seperate_idx_add = self.generator.differentiate(tree, original_expr, input_dimension, "add")
                seperate_idx_mul = self.generator.differentiate(tree, original_expr, input_dimension, "mul")
                seperate_idx_add = self.generator.sample_differentiation(self.rng, seperate_idx_add, sample_prob=0.95)
                add_structure = self.generator.find_expr_structure(self.rng, seperate_idx_add, input_dimension)
                seperate_idx_mul = self.generator.sample_differentiation(self.rng, seperate_idx_mul, sample_prob=0.6)
                mul_structure = self.generator.find_expr_structure(self.rng, seperate_idx_mul, input_dimension)
            
            else:

                add_structure, mul_structure = params.sample_structure_hints.split("|")
                add_structure = [a.split(",") for a in add_structure.split(";")]
                mul_structure = [m.split(",") for m in mul_structure.split(";")]
        
        else:
            add_structure = None
            mul_structure = None

        #consts
        if "consts" in use_hints:
            if params.sample_const_hints == None:

                if variable_consts is not None and (zero_units or self.rng.random() < 0.06):
                    # omit consts
                    const = []
 
                elif variable_consts is not None:
                    # this samples const from sub-exprs
                    const = [
                        [float(_value), np.array(_unit)]
                        for _value, _unit in zip(variable_consts, variable_consts_units)
                    ]
                else:
                    # this samples const occured in expr
                    pt = 0
                    all_nodes = [tree]
                    while pt < len(all_nodes):
                        temp = all_nodes[pt]
                        all_nodes += temp.children
                        pt += 1

                    # ignore pi and integers
                    const_nodes = []
                    for node in all_nodes:
                        try:
                            val = float(node.value)
                            assert int(val) != val
                            const_nodes.append(node)
                        except:
                            pass
                    
                    const_sample_rate = 0.15
                    const = [[n.value, n.unit] for n in const_nodes if self.rng.random() < const_sample_rate]
                    const = [
                        [float(_value), _units] if _value != "pi" else [_value, _units]
                        for _value, _units in const
                    ]
            
            else:
                const = params.sample_const_hints
                if const == "": const = []
                else:
                    const = [c.split(",") for c in const.split(";")]
                    const = [[c[0], c[1:]] for c in const]
        
        else:
            const = None


        # -------------------------------- ENCODE EXPRESSION TREE --------------------------------
        tree_encoded, dim_encoded = self.equation_encoder.encode(tree, zero_units)
        #skeleton_tree, _ = self.generator.function_to_skeleton(tree)
        #skeleton_tree_encoded = self.equation_encoder.encode(skeleton_tree)
        skeleton_tree = None
        skeleton_tree_encoded = None

        assert all(
            [x in self.equation_word2id for x in tree_encoded]
        ), "tree: {}\n encoded: {}".format(tree, tree_encoded)

        # -------------------------------- INTEGRATION ALL PARTS --------------------------------
        X_to_fit, Y_to_fit = [], []
        info = {
            "n_input_points": [],
            "n_unary_ops": [],
            "n_binary_ops": [],
            "d_in": [],
            "d_out": [],
            "input_distribution_type": [],
            "n_centroids": [],
            "expr_idx": [],
        }
        n_input_points = x_to_fit.shape[0]

        if input_length_modulo != -1 and not train:
            indexes_to_keep = np.arange(
                min(input_length_modulo, self.params.max_len),
                self.params.max_len + 1,
                step=input_length_modulo,
            )
        else:
            indexes_to_keep = [n_input_points]

        for idx in indexes_to_keep:
            _x_to_fit = x_to_fit[:idx] if idx > 0 else x_to_fit
            _y_to_fit = y_to_fit[:idx] if idx > 0 else y_to_fit
            X_to_fit.append(_x_to_fit)
            Y_to_fit.append(_y_to_fit)
            info["n_input_points"].append(idx)
            info["n_unary_ops"].append(sum(nb_unary_ops))
            info["n_binary_ops"].append(sum(nb_binary_ops))
            info["d_in"].append(in_out_dimensions["input_dimension"])
            info["d_out"].append(in_out_dimensions["output_dimension"])
            info["input_distribution_type"].append(
                input_distribution_type_to_int[input_distribution_type]
            )
            info["n_centroids"].append(n_centroids)
            info["expr_idx"].append(expr_idx)

        expr = {
            "X_to_fit": X_to_fit,
            "Y_to_fit": Y_to_fit,
            "tree_encoded": tree_encoded,
            "dim_encoded": dim_encoded,
            "skeleton_tree_encoded": skeleton_tree_encoded,
            "tree": tree,
            "skeleton_tree": skeleton_tree,
            "original_expr": original_expr,
            "infos": info,

            "units": units,
            "complexity": complexity,
            "unarys": unary_ops,
            "add_structure": add_structure,
            "mul_structure": mul_structure, 
            "consts": const,
            "real_variables": real_variables,
        }
        for k, (x, y) in predict_datapoints.items():
            expr["x_to_" + k] = x
            expr["y_to_" + k] = y

        return expr, []

    def create_train_iterator(self, task, data_path, params, **args):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")
        dataset = EnvDataset(
            self,
            task,
            train=True,
            skip=self.params.queue_strategy is not None,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
            **args,
        )

        if self.params.queue_strategy is None:
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.collate_reduce_padding(
                dataset.collate_fn,
                key_fn=lambda x: x["infos"]["input_sequence_length"]
                + len(
                    x["tree_encoded"]
                ),  # (x["infos"]["input_sequence_length"], len(x["tree_encoded"])),
                max_size=self.max_size,
            )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 3600),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=collate_fn,
        )

    def create_test_iterator(
        self,
        data_type,
        task,
        data_path,
        batch_size,
        params,
        size,
        input_length_modulo,
        datatype,
        **args,
    ):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            skip=False,
            params=params,
            path=(None if data_path is None else data_path[task][1]),
            size=size,
            type=data_type,
            input_length_modulo=input_length_modulo,
            datatype=datatype,
            **args,
        )

        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--queue_strategy",
            type=str,
            default="uniform_sampling",
            help="in [precompute_batches, uniform_sampling, uniform_sampling_replacement]",
        )

        parser.add_argument("--collate_queue_size", type=int, default=2000)

        parser.add_argument(
            "--use_sympy",
            type=bool_flag,
            default=False,
            help="Whether to use sympy parsing (basic simplification)",
        )
        parser.add_argument(
            "--simplify",
            type=bool_flag,
            default=False,
            help="Whether to use further sympy simplification",
        )
        parser.add_argument(
            "--use_abs",
            type=bool_flag,
            default=False,
            help="Whether to replace log and sqrt by log(abs) and sqrt(abs)",
        )

        # encoding
        parser.add_argument(
            "--operators_to_downsample",
            type=str,
            default="div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--operators_to_not_repeat",
            type=str,
            default="",
            help="Which operator to not repeat",
        )

        parser.add_argument(
            "--max_unary_depth",
            type=int,
            default=6,
            help="Max number of operators inside unary",
        )

        parser.add_argument(
            "--required_operators",
            type=str,
            default="",
            help="Which operator to remove",
        )
        parser.add_argument(
            "--extra_unary_operators",
            type=str,
            default="",
            help="Extra unary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_binary_operators",
            type=str,
            default="",
            help="Extra binary operator to add to data generation",
        )
        parser.add_argument(
            "--extra_constants",
            type=str,
            default=None,
            help="Additional int constants floats instead of ints",
        )

        parser.add_argument("--min_input_dimension", type=int, default=1)
        parser.add_argument("--max_input_dimension", type=int, default=10)
        parser.add_argument("--min_output_dimension", type=int, default=1)
        parser.add_argument("--max_output_dimension", type=int, default=1)
        parser.add_argument(
            "--enforce_dim",
            type=bool,
            default=True,
            help="should we enforce that we get as many examples of each dim ?",
        )

        parser.add_argument(
            "--use_controller",
            type=bool,
            default=True,
            help="should we enforce that we get as many examples of each dim ?",
        )

        parser.add_argument(
            "--float_precision",
            type=int,
            default=3,
            help="Number of digits in the mantissa",
        )
        parser.add_argument(
            "--mantissa_len",
            type=int,
            default=1,
            help="Number of tokens for the mantissa (must be a divisor or float_precision+1)",
        )
        parser.add_argument(
            "--max_exponent", type=int, default=100, help="Maximal order of magnitude"
        )
        parser.add_argument(
            "--max_exponent_prefactor",
            type=int,
            default=1,
            help="Maximal order of magnitude in prefactors",
        )
        parser.add_argument(
            "--max_token_len",
            type=int,
            default=0,
            help="max size of tokenized sentences, 0 is no filtering",
        )
        parser.add_argument(
            "--pad_to_max_dim",
            type=bool,
            default=True,
            help="should we pad inputs to the maximum dimension?",
        )

        # generator
        parser.add_argument(
            "--max_int",
            type=int,
            default=10,
            help="Maximal integer in symbolic expressions",
        )
        parser.add_argument(
            "--min_binary_ops_per_dim",
            type=int,
            default=0,
            help="Min number of binary operators per input dimension",
        )
        parser.add_argument(
            "--max_binary_ops_per_dim",
            type=int,
            default=1,
            help="Max number of binary operators per input dimension",
        )
        parser.add_argument(
            "--max_binary_ops_offset",
            type=int,
            default=4,
            help="Offset for max number of binary operators",
        )
        parser.add_argument(
            "--min_unary_ops", type=int, default=0, help="Min number of unary operators"
        )
        parser.add_argument(
            "--max_unary_ops",
            type=int,
            default=4,
            help="Max number of unary operators",
        )
        parser.add_argument(
            "--min_op_prob",
            type=float,
            default=0.01,
            help="Minimum probability of generating an example with given n_op, for our curriculum strategy",
        )
        
        parser.add_argument(
            "--max_centroids",
            type=int,
            default=10,
            help="Max number of centroids for the input distribution",
        )

        parser.add_argument(
            "--prob_const",
            type=float,
            default=0.0,
            help="Probability to generate integer in leafs",
        )

        parser.add_argument(
            "--reduce_num_constants",
            type=bool,
            default=True,
            help="Use minimal amount of constants in eqs",
        )

        parser.add_argument(
            "--use_skeleton",
            type=bool,
            default=False,
            help="should we use a skeleton rather than functions with constants",
        )

        parser.add_argument(
            "--prob_rand",
            type=float,
            default=0.0,
            help="Probability to generate n in leafs",
        )
        parser.add_argument(
            "--max_trials",
            type=int,
            default=3,
            help="How many trials we have for a given function",
        )

        # evaluation
        parser.add_argument(
            "--n_prediction_points",
            type=int,
            default=200,
            help="number of next terms to predict",
        )


class EnvDataset(Dataset):
    def __init__(
        self,
        env,
        task,
        train,
        params,
        path,
        skip=False,
        size=None,
        type=None,
        input_length_modulo=-1,
        datatype="train",
        **args,
    ):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.skip = skip
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.count = 0
        self.remaining_data = 0
        self.type = type
        self.input_length_modulo = input_length_modulo
        self.params = params
        self.errors = defaultdict(int)

        if "test_env_seed" in args:
            self.test_env_seed = args["test_env_seed"]
        else:
            self.test_env_seed = None
        if "env_info" in args:
            self.env_info = args["env_info"]
        else:
            self.env_info = None

        assert task in FunctionEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0
        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        self.collate_queue: Optional[List] = [] if self.train else None
        self.collate_queue_size = params.collate_queue_size
        self.tokens_per_batch = params.tokens_per_batch

        self.count_queue_num = np.random.randint(999, 1078)
        self.samples = []

        self.datatype = datatype

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def collate_size_fn(self, batch: Dict) -> int:
        if len(batch) == 0:
            return 0
        return len(batch) * max(
            [seq["infos"]["input_sequence_length"] for seq in batch]
        )

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.params.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_reduce_padding(self, collate_fn, key_fn, max_size=None):
        if self.params.queue_strategy == None:
            return collate_fn

        f = self.collate_reduce_padding_uniform

        def wrapper(b):
            try:
                return f(collate_fn=collate_fn, key_fn=key_fn, max_size=max_size,)(b)
            except ZMQNotReady:
                return ZMQNotReadySample()

        return wrapper

    #entrance for train
    def _fill_queue(self, n: int, key_fn):
        """
        Add elements to the queue (fill it entirely if `n == -1`)
        Optionally sort it (if `key_fn` is not `None`)
        Compute statistics
        """
        assert self.train, "Not Implemented"
        assert (
            len(self.collate_queue) <= self.collate_queue_size
        ), "Problem with queue size"

        # number of elements to add
        n = self.collate_queue_size - len(self.collate_queue) if n == -1 else n
        assert n > 0, "n<=0"

        if self.path is None:
            for _ in range(n):
                sample = self.generate_sample()
                self.collate_queue.append(sample)
        else:
            """##TODO
            assert (
                False
            ), "need to finish implementing load dataset, but do not know how to handle read index"
            sample = self.read_sample(index)"""

            while n > 0:
                if len(self.samples) == 0:
                    self.count_queue_num = self.count_queue_num + 1
                    if self.count_queue_num == 1080: self.count_queue_num = 1000
                    with open(f"{self.path}queue_{self.count_queue_num}.pkl", "rb") as fi:
                        self.samples = pickle.load(fi)
                    for i in range(len(self.samples)):
                        #idx = np.random.choice(np.arange(self.samples[i]["infos"]["n_input_points"]), self.params.max_len, replace=False)
                        idx = np.random.randint(0, self.samples[i]["x_to_fit"].shape[0] - self.params.max_len + 1)
                        self.samples[i]["x_to_fit"] = self.samples[i]["x_to_fit"][idx: idx+self.params.max_len]
                        self.samples[i]["y_to_fit"] = self.samples[i]["y_to_fit"][idx: idx+self.params.max_len]
                        self.samples[i]["infos"]["n_input_points"] = self.env.params.max_len
                        self.samples[i]["infos"]["input_sequence_length"] = self.env.params.max_len

                update = self.samples[: n]
                self.samples = self.samples[n:]
                self.collate_queue.extend(update)
                n -= len(update)
                
        # sort sequences
        if key_fn is not None:
            if np.random.random() < 0.5:
                self.collate_queue.sort(key=key_fn, reverse=True)
            else:
                self.collate_queue.sort(key=key_fn, reverse=False)

    def collate_reduce_padding_uniform(self, collate_fn, key_fn, max_size=None):
        """
        Stores a queue of COLLATE_QUEUE_SIZE candidates (created with warm-up).
        When collating, insert into the queue then sort by key_fn.
        Return a random range in collate_queue.
        @param collate_fn: the final collate function to be used
        @param key_fn: how elements should be sorted (input is an item)
        @param size_fn: if a target batch size is wanted, function to compute the size (input is a batch)
        @param max_size: if not None, overwrite params.batch.tokens
        @return: a wrapped collate_fn
        """

        def wrapped_collate(sequences: List):

            if not self.train:
                return collate_fn(sequences)

            # fill queue

            assert all(seq == SKIP_ITEM for seq in sequences)
            assert (
                len(self.collate_queue) < self.collate_queue_size
            ), "Queue size too big, current queue size ({}/{})".format(
                len(self.collate_queue), self.collate_queue_size
            )
            self._fill_queue(n=-1, key_fn=key_fn)
            assert (
                len(self.collate_queue) == self.collate_queue_size
            ), "Fill has not been successful"

            # select random index
            before = self.env.rng.randint(-self.batch_size, len(self.collate_queue))
            before = max(min(before, len(self.collate_queue) - self.batch_size), 0)
            after = self.get_last_seq_id(before, max_size)

            # create batch / remove sampled sequences from the queue
            to_ret = collate_fn(self.collate_queue[before:after])
            self.collate_queue = (
                self.collate_queue[:before] + self.collate_queue[after:]
            )
            return to_ret

        return wrapped_collate

    def get_last_seq_id(self, before: int, max_size: Optional[int]) -> int:
        """
        Return the last sequence ID that would allow to fit according to `size_fn`.
        """
        max_size = self.tokens_per_batch if max_size is None else max_size

        if max_size < 0:
            after = before + self.batch_size
        else:
            after = before
            while (
                after < len(self.collate_queue)
                and self.collate_size_fn(self.collate_queue[before:after]) < max_size
            ):
                after += 1
            # if we exceed `tokens_per_batch`, remove the last element
            size = self.collate_size_fn(self.collate_queue[before:after])
            if size > max_size:
                if after > before + 1:
                    after -= 1
                else:
                    logger.warning(
                        f"Exceeding tokens_per_batch: {size} "
                        f"({after - before} sequences)"
                    )
        return after

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """

        samples = zip_dic(elements)
        info_tensor = {
            info_type: torch.LongTensor(samples["infos"][info_type])
            for info_type in samples["infos"].keys()
        }
        samples["infos"] = info_tensor
        if "input_sequence_length" in samples["infos"]:
            del samples["infos"]["input_sequence_length"]
        errors = copy.deepcopy(self.errors)
        self.errors = defaultdict(int)
        return samples, errors

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.env.rng is not None:
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [worker_id, self.params.global_rank, self.env_base_seed]
            if self.env_info is not None:
                seed += [self.env_info]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{seed} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            seed = [
                worker_id,
                self.params.global_rank,
                self.test_env_seed if "valid" in self.type else 0,
            ]
            self.env.rng = np.random.RandomState(seed)
            logger.info(
                "Initialized {} generator, with seed {} (random state: {})".format(
                    self.type, seed, self.env.rng
                )
            )

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0), "issue in worker id"
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    #entrance for test
    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                sample = self.generate_sample()
                return sample
        else:
            if self.train and self.skip:
                return SKIP_ITEM
            else:
                return self.samples[index]
                #return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index

        def str_list_to_float_array(lst):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    lst[i][j] = float(lst[i][j])
            return np.array(lst)

        x = copy.deepcopy(self.data[idx])
        x["x_to_fit"] = str_list_to_float_array(x["x_to_fit"])
        x["y_to_fit"] = str_list_to_float_array(x["y_to_fit"])
        x["x_to_predict"] = str_list_to_float_array(x["x_to_predict"])
        x["y_to_predict"] = str_list_to_float_array(x["y_to_predict"])
        x["tree"] = self.env.equation_encoder.decode(x["tree"].split(","))
        x["tree_encoded"] = self.env.equation_encoder.encode(x["tree"])
        infos = {}

        for col in x.keys():
            if col not in [
                "x_to_fit",
                "y_to_fit",
                "x_to_predict",
                "y_to_predict",
                "tree",
                "tree_encoded",
            ]:
                infos[col] = int(x[col])
        x["infos"] = infos
        for k in infos.keys():
            del x[k]
        return x

    def generate_sample(self):
        """
        Generate a sample.
        """

        if self.remaining_data == 0:
            self.expr, errors = self.env.gen_expr(
                self.train, 
                input_length_modulo=self.input_length_modulo,
                datatype=self.datatype
            )
            for error, count in errors.items():
                self.errors[error] += count

            self.remaining_data = len(self.expr["X_to_fit"])

        self.remaining_data -= 1
        x_to_fit = self.expr["X_to_fit"][-self.remaining_data]
        y_to_fit = self.expr["Y_to_fit"][-self.remaining_data]
        sample = copy.deepcopy(self.expr)
        sample["x_to_fit"] = x_to_fit
        sample["y_to_fit"] = y_to_fit
        del sample["X_to_fit"]
        del sample["Y_to_fit"]
        sample["infos"] = select_dico_index(sample["infos"], -self.remaining_data)
        sequence = []
        for n in range(sample["infos"]["n_input_points"]):
            sequence.append([sample["x_to_fit"][n], sample["y_to_fit"][n]])
        sample["infos"]["input_sequence_length"] = self.env.get_length_after_batching(
            [sequence]
        )[0].item()
        #if sample["infos"]["input_sequence_length"] > self.params.tokens_per_batch:
            # print(sample["infos"]["input_sequence_length"],  self.params.tokens_per_batch)
            #return self.generate_sample()
        self.count += 1
        return sample


def select_dico_index(dico, idx):
    new_dico = {}
    for k in dico.keys():
        new_dico[k] = dico[k][idx]
    return new_dico
