# seperate_main.py
# seperate the whole expr into some small sub-exprs

import json
import sympy as sp
import numpy as np
import pandas as pd
import tqdm
import scipy
from collections import defaultdict
import traceback
import json
import re

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parsers import get_parser
import symbolicregression as symbolicregression
from symbolicregression.envs import build_env

import warnings
warnings.filterwarnings("ignore", message="masked_fill_ received a mask with dtype torch.uint8")
warnings.filterwarnings("ignore", message="We've integrated functorch into PyTorch.*")
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated:*")

def seperation_protect(expr, variables):
    protect = []
    for k,v in {
        "sqrt(1 - v**2/c**2)": ["v", "c"],
        "sqrt(1 - u**2/c**2)": ["u", "c"],
        "sqrt(1 - v1**2/c**2))": ["v1", "c"],
        "sqrt((c**2 - v**2)/c**2)": ["v", "c"],
        "sqrt((c**2 - u**2)/c**2)": ["u", "c"],
        "sqrt((c**2 - v1**2)/c**2)": ["v1", "c"],
    }.items():
        if k in expr:
            idx1 = variables.index(v[0])
            idx2 = variables.index(v[1])

            protect = [idx1, idx2]
            break

    return protect


if __name__ == "__main__":

    #init exp
    np.random.seed(2024)
    parser = get_parser()
    params = parser.parse_args()
    env = build_env(params)
    generator = env.generator

    for datatype in ["train", "valid"]:

        with open(f"./data/exprs_{datatype}.json", "r", encoding="utf-8") as fi:
            context = json.load(fi)
        new_context = []

        independenet_sample_time = 2

        pbar = tqdm.tqdm(total = len(context))

        for i,expr in enumerate(context):
            pbar.update(1)

            infix = expr

            try:
                node, variables = generator.infix_to_node(infix, sp_parse=False)
            except:
                message = traceback.format_exc()
                continue

            total_variance = []

            add_seperation = generator.differentiate(node, expr, len(variables), structure_type="add")
            mul_seperation = generator.differentiate(node, expr, len(variables), structure_type="mul")

            idx_protect = seperation_protect(expr, variables)

            #try total sub-structure
            _add_seperation = generator.sample_differentiation(np.random, add_seperation, sample_prob=1)
            _mul_seperation = generator.sample_differentiation(np.random, mul_seperation, sample_prob=1)

            add_structure = generator.find_expr_structure(None, _add_seperation, len(variables))
            mul_structure = generator.find_expr_structure(None, _mul_seperation, len(variables))

            for struc in [add_structure, mul_structure]:
                for _struc in struc:
                    if str(_struc) not in total_variance and len(_struc) != len(variables):
                        if idx_protect: 
                            if (idx_protect[0] in _struc and idx_protect[1] in _struc)\
                                or (idx_protect[0] not in _struc and idx_protect[1] not in _struc):
                                total_variance.append((str(_struc)))
                        else:
                            total_variance.append((str(_struc)))

            #sampling sub-structure
            for _ in range(independenet_sample_time):

                _add_seperation = generator.sample_differentiation(np.random, add_seperation, sample_prob=0.7)
                _mul_seperation = generator.sample_differentiation(np.random, mul_seperation, sample_prob=0.7)

                add_structure = generator.find_expr_structure(None, _add_seperation, len(variables))
                mul_structure = generator.find_expr_structure(None, _mul_seperation, len(variables))

                for struc in [add_structure, mul_structure]:
                    for _struc in struc:
                        if str(_struc) not in total_variance and len(_struc) != len(variables):
                            if idx_protect: 
                                if (idx_protect[0] in _struc and idx_protect[1] in _struc)\
                                    or (idx_protect[0] not in _struc and idx_protect[1] not in _struc):
                                    total_variance.append((str(_struc)))
                            else:
                                total_variance.append((str(_struc)))

            current_context = []
            for variance_struc in total_variance:

                variance_struc = list(map(int, variance_struc[1:-1].split(" ")))

                for trial in range(3):
                    tree = node.copy()
                    fixed_x = np.random.uniform(1, 3, (len(variables)))
                    transition_dic = {
                        f"x_{i}": fixed_x[i]
                        for i in range(len(variables))
                    }
                    for i in variance_struc:
                        transition_dic[f"x_{i}"] = variables[i]
                    transition_dic["pi"] = 3.14159265
                    transition_dic["E"] = 2.7182818

                    generator.apply_transition(tree, transition_dic)

                    generator.simplify_tree(tree)

                    #new_variables = [v for v in variables if v in tree.prefix().split(",")]

                    #randomly add dummy variables
                    #for _v in variables:
                    #    if _v not in new_variables and np.random.random() < 0.15:
                    #        new_variables.insert(np.random.randint(0, len(new_variables)+1) ,_v)

                    transition_dic = {
                        k:v for k, v in {
                            "add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"
                        }.items()
                    }
                    generator.apply_transition(tree, transition_dic)

                    units = []
                    stack = [tree]
                    while stack:
                        temp = stack.pop(0)
                        units.append(
                            list(map(int, temp.unit.tolist()))
                        )
                        stack = temp.children + stack

                    variables_units = [
                        list(map(
                            int, generator.physical_units_dic[v].tolist()
                        ))
                        for v in variables
                    ]

                    consts = [
                        f"{fixed_x[i]:.4g}"
                        for i in range(len(variables)) if i not in variance_struc
                    ]

                    consts_units = [
                        variables_units[i]
                        for i in range(len(variables)) if i not in variance_struc
                    ]

                    new_expr = re.sub(r"\b\d+\.\d+\b", lambda x: f"{float(x.group()):.4g}", str(tree))
                    new_expr = new_expr.replace("* 1/", "/")

                    if "nan" in new_expr:
                        a = 1
                        continue
                    else:
                        break
                
                if "nan" in new_expr:
                    a = 1
                    continue

                current_context.append({
                    "expr": new_expr,
                    #"variables": new_variables,
                    "units": units,
                    "consts": consts,
                    "c_units": consts_units,
                })

                if "E" in new_expr and "E_n" not in new_expr and "Ef" not in new_expr:
                    a = 1

                #test
                test_node, _ = generator.infix_to_node(
                        new_expr, 
                        rng=np.random, 
                        #variables=new_variables,
                        label_units=False,
                        random_variables_seq=False,
                        sp_parse=False
                    )
                generator.label_units_with_consts(test_node, units)

                a = 1

            #append or extend?
            #depending on how to sample sub-exprs?
            new_context.extend(current_context)
            a = 1

        pbar.close()

        with open(f"./data/exprs_seperated_{datatype}.json", "w", encoding="utf-8") as fi:
            json.dump(new_context, fi)
        


