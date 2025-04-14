import numpy as np
import copy
import sympy as sp
import torch
import os
import re
from sklearn.metrics import r2_score
from scipy.optimize import minimize

from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from symbolicregression.model.model_wrapper import ModelWrapper

from Oracle.oracle import Oracle
from symbolicregression.MCTS.mcts import MCTS
from symbolicregression.GA.ga import GeneticProgramming, TreeGenerator

import warnings
warnings.filterwarnings("ignore", message="masked_fill_ received a mask with dtype torch.uint8")
warnings.filterwarnings("ignore", message="We've integrated functorch into PyTorch.*")
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated:*")


class PhyReg():
    
    def __init__(self, 
                path, 
                max_len=None,
                refinement_strategy=None,
                device=None,
                ):
        
        model = torch.load(path)
        params = model['params']

        params.rescale = False

        if max_len is not None:
            assert isinstance(max_len, int) and max_len > 0
            params.max_len = max_len
            params.max_input_points = max_len
        if refinement_strategy is not None:
            assert isinstance(refinement_strategy, str)
            params.refinement_strategy = refinement_strategy
        if device is not None:
            assert "cuda" in device
            params.device = device

        env = build_env(params)
        modules = build_modules(env, params)
        oracle = Oracle(env, env.generator, params)

        self.params = params
        self.env = env
        self.modules = modules
        self.oracle = oracle

        embedder = modules["embedder"]
        encoder = modules["encoder"]
        decoder = modules["decoder"]

        embedder.load_state_dict(model['embedder'])
        encoder.load_state_dict(model['encoder'])
        decoder.load_state_dict(model['decoder'])

        embedder.eval()
        encoder.eval()
        decoder.eval()

        self.mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        self.dstr = SymbolicTransformerRegressor(
            model=self.mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        #env.rng = np.random.RandomState(params.seed)

    def save(self, path):
        save_dict = {
            'embedder': self.mw.embedder.state_dict(),
            'encoder': self.mw.encoder.state_dict(),
            'decoder': self.mw.decoder.state_dict(),
            'params': self.params
        }
        torch.save(save_dict, path)

    @staticmethod
    def encode_units(units):
        """
        encode "kg0m1s-1T0V0" into [0,1,-1,0,0]
        """
        if units is None:
            return np.array([0., 0, 0, 0, 0])
        else:
            return np.array([
                float((units.split("m"))[0][2:]),
                float((((units.split("m"))[1]).split("s"))[0]),
                float((((units.split("s"))[1]).split("T"))[0]),
                float((((units.split("T"))[1]).split("V"))[0]),
                float((units.split("V"))[1]),
            ])


    @staticmethod
    def decode_units(units):
        """
        decode [0,1,-1,0,0] into "kg0m1s-1T0V0"
        """
        return f"kg{int(units[0])}m{int(units[1])}s{int(units[2])}T{int(units[3])}V{int(units[4])}"

    @staticmethod
    def eval_metric(y_true, y_pred, metric="mse"):
        y_true = y_true.reshape((-1))
        y_pred = y_pred.reshape((-1))
        if metric == "mse":
            return np.mean((y_pred - y_true) ** 2)
        elif metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mae":
            return np.mean(np.abs(y_pred - y_true))  
        elif metric == "corr":
            #pearson corelation coefficient
            return np.corrcoef(y_true, y_pred)[0][1]
        else:
            raise ValueError

    @staticmethod
    def _const_optimize(expr, xy, dimension, x0, method="BFGS"):
        (x, y) = xy
        x_variables = [sp.Symbol('x_{}'.format(i)) for i in range(x.shape[1])]
        c_variables = [sp.Symbol('c_{}'.format(i)) for i in range(dimension)]
        func = sp.lambdify(x_variables+c_variables, expr, ["numpy", {"sigmoid": (lambda x: 1 / (1 + np.exp(-x)))}])
        def loss(c, x, y):
            pred = np.array([func(*x.T, *c)])
            return np.mean(np.square(pred.reshape((-1)) - y.reshape((-1))))
        try:
            res = minimize(loss, x0=x0, args=(x, y), method=method)
            return res.x
        except:
            #import traceback
            #message = traceback.format_exc()
            return x0

    def fit(self, x, y, 
            units=None, 
            complexitys=None, 
            unarys=None, 
            consts=None,
            use_Divide=True,
            use_MCTS=True,
            use_GP=True,
            use_const_optimization=False,
            save_oracle_model=False,
            oracle_name="demo",
            oracle_file=None,
            use_pysr_init=False,
            oracle_lr=None,
            oracle_bs=None,
            oracle_epoch=None,
            use_seperate_type=None,
            fixed_x_range=None,
            variable_scale=None,
            verbose=True):
        """
            Parameters
            ----------
            x : np.ndarry | list [np.ndarray]
                input datapoint of x;
                np.ndarray with shape of (N, Nv) for one formula;
                list with m np.ndarray of shape of (Ni, Nvi) for m independent formula
            

            y : np.ndarry | list [np.ndarray]
                input datapoint of y;
                np.ndarray with shape of (N, 1) for one formula;
                list with m np.ndarray of shape of (Ni, 1) for m independent formula


            units: list [str] | list [list [str]]
                physics units of input/output variables;
                use 5 unit basis: kilogram(kg), meter(m), seconds(s), Temperature(T), Volt(V);
                None indicate it is unknown;

                E.g. the format should be ["kg0m1s-1T0V0", ...] for one formula;
                and [["kg0m1s-1T0V0", ...], ...] for m independent formula;


            complexitys: int | list [int]
                expected complexity of target formulas;
                None indicate it is unknown or unlimited;

                E.g. the format should be 10 for one formula;
                and [10, ...] for m independent formula;


            unarys: list [str] | list [list [int]]
                possible unarys used in the target formulas;
                None indicate it is unknown or unlimited;

                E.g. the format should be ["exp", "sin", ...] for one formula;
                and [["exp", "sin", ...], ...] for m independent formula;


            consts: list | list [list]
                possible constants and its units used in the target formulas;
                None indicate it is unknown or unlimited;

                E.g. the format should be [["3.1415", "kg0m0s0T0V0", ["1.5", "kg0m1s-1T0V0"]], ...] for one formula;
                and [[["3.1415", "kg0m0s0T0V0", ["1.5", "kg0m1s-1T0V0"]], ...], ...] for m independent formula;


            use_Divide: Bool
                Do you want to apply Divide-and-Conquer strategy before and after End-to-End prediction?
                Default to be True


            use_MCTS: Bool
                Do you want to apply Monte-Carlo-Tree-Search refinement after End-to-End prediction?
                Default to be True


            use_GP: Bool
                Do you want to apply Genetic-Programming refinement after End-to-End prediction?
                Default to be True


            save_oracle_model:


            verbose: Bool
                Whether to print detailed information or progress during the execution of the function.  
                Default to be True
        """

        self.best_gens = None
        self.best_gens_mcts = None
        self.best_gens_gp = None
        self.best_gens_refined = None

        ################### data preprocess ###################

        if isinstance(x, np.ndarray):
            x = [x]; y = [y]
            units = [units]; complexitys = [complexitys]
            unarys = [unarys]; consts = [consts]
        for i in range(len(y)):
            y[i] = y[i].reshape((-1, 1))

        num_xy = len(x)

        if units is None: units = [None for _ in range(num_xy)]
        if complexitys is None: complexitys = [None for _ in range(num_xy)]
        if unarys is None: unarys = [None for _ in range(num_xy)]
        if consts is None: consts = [None for _ in range(num_xy)]

        ################### check validity of data ###################

        # check variable numbers = units numbers - 1
        for i in range(num_xy):
            if units[i] is not None:
                assert len(units[i]) == (x[i].shape)[1] + 1, f"Formula {i+1}-th, variable numbers={(x[i].shape)[1]}, units numbers={len(units[i])}, invalid!"

        # check the consistency of shape of x and y
        assert len(y) == len(x), f"totally {len(x)} of datapoint of x, while {len(y)} of datapoint of y, invalid!"
        for i in range(num_xy):
            assert (x[i].shape)[0] == (y[i].shape)[0], f"Formula {i+1}-th, number of datapoint of x={(x[i].shape)[0]}, number of datapoint of={(y[i].shape)[0]}, invalid!"


        ################### hints preprocess ###################

        hints = []
        for used_hints in self.params.use_hints.split(","):

            if used_hints == "units":
                enc_units = []
                for i in range(num_xy):
                    if units[i] is None:
                        enc_units.append([
                            self.encode_units(None) 
                            for j in range( (x[i].shape)[1]+1 )
                        ])
                    else:
                        enc_units.append([
                            self.encode_units(units[i][j]) 
                            for j in range( (x[i].shape)[1]+1 )
                        ])
                hints.append(enc_units)

            if used_hints == "complexity":
                
                enc_comp = []
                for comp in complexitys:
                    if comp is None:    enc_comp.append([0])
                    elif comp <= 12:    enc_comp.append(["simple"])
                    elif comp <= 18:    enc_comp.append(["middle"])
                    else:               enc_comp.append(["hard"])
                hints.append(enc_comp)

            if used_hints == "unarys":

                enc_unary = []
                for unary in unarys:
                    if unary is None: enc_unary.append([])
                    else: enc_unary.append(unary)
                hints.append(enc_unary)
            
            if used_hints == "consts":

                enc_consts = []
                for const in consts:
                    if const is None: enc_consts.append([])
                    else:
                        enc_consts.append([
                            [float(c), self.encode_units(u)]
                            for c, u in const
                        ])
                hints.append(enc_consts)
     
        ################### Divide-and-Conquer ###################

        if use_Divide:

            if verbose: print("Training oracle Newral Network...", end="\n")

            if oracle_lr is None: oracle_lr = 0.005
            if oracle_bs is None: oracle_bs = min([int(len(x[i])/3) for i in range(num_xy)])
            if oracle_epoch is None: oracle_epoch = 400
            if use_seperate_type is None: use_seperate_type = self.params.oracle_seperation_type.split(",")

            res_x, res_y, res_hints = self.oracle.oracle_fit(
                x, y, np.arange(len(x)), hints,
                save_model = save_oracle_model,
                lr=oracle_lr, batch_size=oracle_bs,
                epochs=oracle_epoch,
                name=oracle_name, 
                oracle_file=oracle_file,
                fixed_x_range=fixed_x_range,
                use_seperate_type=use_seperate_type
            )

            total_x = [
                *[_x[:self.params.max_input_points*max(1, self.params.max_number_bags)] for _x in x], *res_x
            ]
            total_y = [
                *[_y[:self.params.max_input_points*max(1, self.params.max_number_bags)] for _y in y], *res_y
            ]
            total_hints = [
                [*hint, *res_hint] for hint, res_hint in zip(hints, res_hints)
            ]

            if verbose: print("Generating formula through End-to-End...", end="\n")

            if variable_scale is not None:
                if isinstance(variable_scale[0], int):
                    total_y[variable_scale[0]] *= variable_scale[1]

            self.dstr.fit(total_x, total_y, total_hints, verbose=verbose, refinement_types=["id"])

            best_gens_noref = copy.deepcopy(
                self.dstr.retrieve_tree(
                    refinement_type="NoRef", dataset_idx=-1, with_infos=True
                )
            )
            best_gens_bfgs = copy.deepcopy(
                self.dstr.retrieve_tree(
                    refinement_type="BFGS", dataset_idx=-1, with_infos=True
                )
            )
            
            if verbose: print("Back aggregating formulas...", end="\n")

            best_gens_oracle = self.oracle.reverse(
                best_gens_noref[:len(x)], 
                best_gens_noref[len(x):],
                eliminate=True,
            )

            best_gens = copy.deepcopy(best_gens_oracle)[0::2]

            if verbose:
                self.express_best_gens(best_gens)
            self.best_gens = best_gens
        
        else:

            if verbose: print("Generating formula through End-to-End...", end="\n")

            self.dstr.fit(x, y, hints, verbose=verbose, refinement_types=["id"])

            best_gens_noref = copy.deepcopy(
                self.dstr.retrieve_tree(
                    refinement_type="NoRef", dataset_idx=-1, with_infos=True
                )
            )
            best_gens_bfgs = copy.deepcopy(
                self.dstr.retrieve_tree(
                    refinement_type="BFGS", dataset_idx=-1, with_infos=True
                )
            )

            best_gens = best_gens_bfgs

            for i in range(len(best_gens)):
                if "_mse" not in best_gens[i]:
                    expr = str(best_gens[i]["predicted_tree"])
                    for k,v in {
                        "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
                    }.items():
                        expr = expr.replace(k, v)
                    x_variables = [sp.Symbol('x_{}'.format(t)) for t in range(x[i].shape[1])]
                    f = sp.lambdify(x_variables, expr, "numpy")
                    pred = f(*x[i].T)
                    best_gens[i]["_mse"] = self.eval_metric(y[i], pred)

            self.best_gens = best_gens
            if verbose:
                self.express_best_gens(best_gens)

        self.best_gens_noref = best_gens_noref
        self.best_gens_bfgs = best_gens_bfgs
            

        ################### MCTS ###################

        if use_MCTS:

            best_gens_mcts = copy.deepcopy(best_gens)
            if verbose: print("Refining formula through MCTS...", end="\n")

            for i, best_gen_mcts in enumerate(best_gens_mcts):
                
                num_variables = (x[i].shape)[1]
                base_grammar = {
                    "O": ["A"],
                    "A": ["(A+A)", "(A-A)", "(A*A)", "(A/A)", "exp(A)", "cos(B)", "sin(B)", "B"] + [f"x_{i}" for i in range(num_variables)],
                    "B": ["(B+B)", "(B-B)", "1", "pi"] + [f"(x_{i})**2" for i in range(num_variables)] + [f"x_{i}" for i in range(num_variables)]
                }
                non_terminal = ["O", "A", "B"]

                mcts = MCTS(
                    self.params, self.env, 
                    base_grammar, non_terminal,
                    mcts_print_freq=self.params.mcts_print_freq,
                    search_round=self.params.search_round,
                    num_simulations=self.params.num_simulations,
                    mcts_search_type="max",
                    early_stop=self.params.mcts_early_stop,
                )

                mcts.extract_subexpr(best_gen_mcts["predicted_tree"], update=True)
                target_str, target_length = mcts.extract_target_str(best_gen_mcts["predicted_tree"])
                solutions = mcts.extract_original_solution(best_gen_mcts["predicted_tree"], (x[i], y[i]))

                for _str, _len in zip(target_str, target_length):
                    if len(solutions) > 0:
                        mse = mcts.mse(solutions[-1][1], (x[i], y[i]))
                        if solutions[-1][2] > self.params.mcts_early_stop or mse < 1e-10:
                            break
                    productions = [["O", _str]]
                    nt = [c for c in _str if c == "A" or c == "B"]
                    solution = mcts.search(
                        (x[i], y[i]), productions, nt, 
                        verbose=verbose, 
                        max_time=max(5, 25-_len)
                    )
                    solutions.append(solution)

                solutions.sort(key=lambda x: x[2])
                expr_str = solutions[-1][1]
                try:
                    node, _ = self.env.generator.infix_to_node(
                        expr_str.replace("pow", "**"),
                        sp_parse=True, allow_pow=True, label_units=False
                    )
                except:
                    node, _ = self.env.generator.infix_to_node(
                        "0", sp_parse=False, label_units=False
                    )
                best_gen_mcts["predicted_tree"] = node
            
            if verbose:
                self.express_best_gens(best_gens_mcts)
            self.best_gens_mcts = best_gens_mcts

        ################### GP ###################

        if use_GP:

            best_gens_gp = copy.deepcopy(best_gens)
            if verbose: print("Refining formula through GP...", end="\n")
            
            # if we use D&C strategy
            # then we use all the results from oracle model to init GP
            if use_Divide:
                gp_exprs = [
                    e["predicted_tree"] for e in self.oracle.oracle_exprs
                ] + [
                    e["predicted_tree"] for e in best_gens
                ]
                for i in range(len(gp_exprs)):
                    gp_exprs[i] = str(gp_exprs[i])
                    for k,v in {
                        "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
                    }.items():
                        gp_exprs[i] = gp_exprs[i].replace(k, v)
                    if "nan" in gp_exprs[i] or "None" in gp_exprs[i]:
                        gp_exprs[i] = "0"
                gp_exprs = [
                    gp_exprs[v[0]: v[-1]+1] + [gp_exprs[len(self.oracle.oracle_exprs) + i]]
                    for i, (k,v) in enumerate(self.oracle.original_expr_idx_to_oracle_expr_idx.items())
                ]

            # if we do not use D&C strategy
            # then we use all the results from e2e model to init GP
            else:
                gp_exprs = [
                    e["predicted_tree"] for e in best_gens
                ]
                for i in range(len(gp_exprs)):
                    gp_exprs[i] = str(gp_exprs[i])
                    for k,v in {
                        "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
                    }.items():
                        gp_exprs[i] = gp_exprs[i].replace(k, v)
                    if "nan" in gp_exprs[i] or "None" in gp_exprs[i]:
                        gp_exprs[i] = "0"
                gp_exprs = [
                    [gp_exprs[i]] for i in range(len(gp_exprs))
                ]


            for i, (best_gen_gp, gp_expr) in enumerate(zip(best_gens_gp, gp_exprs)):

                num_variables = (x[i].shape)[1]

                if best_gen_gp["_mse"] < 1e-8:
                    continue
                
                if use_pysr_init:

                    # pysr warm up
                    from pysr import PySRRegressor
                    warnings.filterwarnings("ignore", message="`torch` was loaded before the Julia instance started.*")
                    warnings.filterwarnings("ignore", message="Your system's Python library is static (e.g., conda)*")
                    if verbose: print("Initializing through PySR...", end="\n")
                    pysr = PySRRegressor(
                        model_selection="best",  # Result is mix of simplicity+accuracy
                        niterations=40,
                        binary_operators=["*", "+", "-", "/"],
                        unary_operators=["square", "cube", "exp", "log", "sin", "cos", "tan", "sqrt"],
                        extra_sympy_mappings={"inv": lambda x: 1 / x},
                        loss="loss(x, y) = (x - y)^2",
                        verbosity=verbose
                    )
                    pysr.fit(x[i], y[i])
                    expr_str = str(pysr.sympy())
                    transition_dic = {
                        f"x{j}": f"x_{j}" for j in range(num_variables)
                    }
                    for k,v in transition_dic.items():
                        expr_str = expr_str.replace(k, v)
                    gp_expr = gp_expr + [expr_str]

                    # clear 
                    cnt = 0
                    for root, _, files in os.walk(os.getcwd()):
                        for file_name in files:
                            if file_name.startswith('hall_of_fame'):
                                file_path = os.path.join(root, file_name)
                                try:
                                    os.remove(file_path)
                                    cnt += 1
                                    #print(f"Deleted: {file_path}")
                                except OSError as e:
                                    print(f"Error deleting {file_path}: {e}")
                    #print(f"total remove {cnt} files")
                
                # mixture-heuristic gp search
                assert num_variables == (x[i].shape)[1], f"{num_variables}, {(x[i].shape)[1]}"
                treeGenerator = TreeGenerator(self.params)
                gp = GeneticProgramming(treeGenerator, self.params, self.oracle, max_attemp=5,)
                best_of_all = gp.run(
                    self.env, num_variables, 
                    (x[i], y[i]), exprs=gp_expr, 
                    verbose=verbose
                )
                #print(best_of_all)
                #print(best_of_all.best(), type(best_of_all.best()))
                #print()

                node = best_of_all.best()
                best_gen_gp["predicted_tree"] = node

            if verbose:
                self.express_best_gens(best_gens_gp)
            self.best_gens_gp = best_gens_gp

        ################### Constant Optimization ###################

        if use_const_optimization:
            
            best_gens_refined = copy.deepcopy(best_gens)
            if use_MCTS: best_gens_refined = copy.deepcopy(best_gens_mcts)
            if use_GP: best_gens_refined = copy.deepcopy(best_gens_gp)
            if verbose: print("Refining constants...", end="\n")

            for i in range(num_xy):

                best_expr = ""
                best_mse = 1e100

                expr = str(best_gens_refined[i]["predicted_tree"])
                for k,v in {
                    "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
                }.items():
                    expr = expr.replace(k, v)

                consts = re.findall(r"[+-]?\b\d+\.\d+\b", expr)
                for t,c in enumerate(consts):
                    expr = expr.replace(c, f"c_{t}", 1)
                consts = np.array(list(map(float, consts)))

                for trial in range(self.params.num_bfgs):
                    
                    new_consts = copy.deepcopy(consts)
                    if trial > 0:new_consts = new_consts + np.random.normal(0, 1, new_consts.shape)
                    new_expr = copy.deepcopy(expr)

                    new_consts = self._const_optimize(new_expr, (x[i], y[i]), len(consts), new_consts)
                    for t,c in enumerate(new_consts):
                        new_expr = new_expr.replace(f"c_{t}", str(c), 1)
                    
                    x_variables = [sp.Symbol('x_{}'.format(t)) for t in range(x[i].shape[1])]
                    f = sp.lambdify(x_variables, new_expr, "numpy")
                    pred = f(*x[i].T)
                    mse = self.eval_metric(y[i], pred, metric="mse")

                    if mse < best_mse:
                        best_mse = mse
                        best_expr = new_expr

                best_gens_refined[i]["predicted_tree"] = best_expr
                best_gens_refined[i]["_mse"] = best_mse

            if verbose:
                self.express_best_gens(best_gens_refined)
            self.best_gens_refined = best_gens_refined

    def constant_optimization(self, best_gens, x, y, sigma=1):

        best_gens_refined = copy.deepcopy(best_gens)

        for i in range(len(x)):

            best_expr = ""
            best_mse = 1e100

            expr = str(best_gens_refined[i]["predicted_tree"])
            for k,v in {
                "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
            }.items():
                expr = expr.replace(k, v)

            consts = re.findall(r"[+-]?\b\d+\.\d+\b", expr)
            for t,c in enumerate(consts):
                expr = expr.replace(c, f"c_{t}", 1)
            consts = np.array(list(map(float, consts)))

            for trial in range(self.params.num_bfgs):
                
                new_consts = copy.deepcopy(consts)
                if trial > 0:new_consts = new_consts + np.random.normal(0, sigma, new_consts.shape)
                new_expr = copy.deepcopy(expr)

                new_consts = self._const_optimize(new_expr, (x[i], y[i]), len(consts), new_consts)
                for t,c in enumerate(new_consts):
                    new_expr = new_expr.replace(f"c_{t}", str(c), 1)
                
                x_variables = [sp.Symbol('x_{}'.format(t)) for t in range(x[i].shape[1])]
                f = sp.lambdify(x_variables, new_expr, "numpy")
                pred = f(*x[i].T)
                mse = self.eval_metric(y[i], pred, metric="mse")

                if mse < best_mse:
                    best_mse = mse
                    best_expr = new_expr

            best_gens_refined[i]["predicted_tree"] = best_expr
            best_gens_refined[i]["_mse"] = best_mse

        return best_gens_refined


    def express_best_gens(self, best_gens, use_sp=False):
        for i in range(len(best_gens)):
            expr = str(best_gens[i]["predicted_tree"])
            mse = str(best_gens[i]["_mse"])
            for k,v in {
                "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
            }.items():
                expr = expr.replace(k, v)
            if use_sp:
                expr = str(sp.parse_expr(expr).evalf())
            print(f"idx : {i}")
            print(f"expr: {expr}")
            print(f"mse : {mse}")
            print()

    def express_skeleton(self, best_gens, use_sp=False):
        for i in range(len(best_gens)):
            expr = str(best_gens[i]["predicted_tree"])
            for k,v in {
                "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"
            }.items():
                expr = expr.replace(k, v)
            if use_sp:
                expr = str(sp.parse_expr(expr).evalf())
            consts = re.findall(r'-?\d+\.\d+e[+-]?\d+|-?\d+\.\d+|-?\d+e[+-]?\d+', expr)
            for t,c in enumerate(consts):
                expr = expr.replace(c, f"C_{t}", 1)
            consts = " ".join([str(round(float(c), 3)) for c in consts])
            print(f"idx          : {i}")
            print(f"expr skeleton: {expr}")
            print(f"constants    : {consts}")
            print()
