#given generated datasets(exprs)
#preprocess to dismiss those ilegal

import json
import sympy as sp
import numpy as np
import pandas as pd
import tqdm
import scipy
from collections import defaultdict
import traceback
import json

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parsers import get_parser
import symbolicregression as symbolicregression
from symbolicregression.envs import build_env

import warnings
warnings.filterwarnings("ignore", message="masked_fill_ received a mask with dtype torch.uint8")
warnings.filterwarnings("ignore", message="We've integrated functorch into PyTorch.*")
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated:*")


def remove_duplica(context, generator):
    exprs = defaultdict(int)
    for expr in context:
        exprs[expr] += 1
    return list(exprs.keys())

def _cal_dimension(expr, generator):
    if isinstance(expr, str):
        expr = sp.parse_expr(expr, local_dict=generator.locals)

    flag = True
    if not expr.args:
        if isinstance(expr, sp.core.numbers.Number):
            return np.zeros((5)), True
        return np.array(generator.physical_units_dic[str(expr)]), True

    if isinstance(expr, sp.core.add.Add):
        ans, f = _cal_dimension(expr.args[0], generator)
        flag = flag and f
        for i in range(1, len(expr.args)):
            d, f = _cal_dimension(expr.args[i], generator)
            flag = flag and f
            if not all(ans == d):
                flag = False
                break

    elif isinstance(expr, sp.core.mul.Mul):
        ans = np.zeros((5))
        for sub in expr.args:
            d, f = _cal_dimension(sub, generator)
            ans += d
            flag = flag and f

    elif isinstance(expr, sp.core.power.Pow):
        d1, f1 = _cal_dimension(expr.args[0], generator)
        d2, f2 = _cal_dimension(expr.args[1], generator)

        try:
            return d1 * int(expr.args[1]), f1 and f2
        except:
            #TODO:
            #we may make some refinement here?
            #but now we assume those expr with non-const in power are wrong
            return np.zeros((5)), False
    
    elif isinstance(expr, (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh)):
        assert len(expr.args)==1
        d, f = _cal_dimension(expr.args[0], generator)
        if all(d == 0):
            return d, f
        else:
            return np.zeros((5)), False    
    else:
        assert False
        print("Error in type!", type(expr))
        return np.zeros((5)), False 

    return ans, flag

def replace_dimension_failure(expr, generator, op):
    assert op in ["exp", "log", "sin", "cos"]
    sp_expr = sp.parse_expr(expr, local_dict=generator.locals)

    sub_variables = set()
    stack = [sp_expr]
    while stack:
        temp = stack.pop(0)
        if isinstance(temp, getattr(sp, op)):
            sub_variables = sub_variables.union(temp.free_symbols)
        stack += temp.args

    sp_expr = sp_expr.subs({v: sp.symbols(f"x_{i}") for i,v in enumerate(sub_variables)})
    return str(sp_expr)

def preprocess(context, generator):

    failure_length = []
    failure_complete = []
    failure_legalOperator = []
    failure_inequality = []
    failure_variable = []
    failure_sympy = []
    failure_const = []
    failure_dimension = []

    pbar = tqdm.tqdm(total=len(context))

    #check validity
    for i,expr in enumerate(context):
        pbar.update(1)
        
        #check if it is too long
        #if expr[-1:] != "\n":
        #    failure_length.append(i)
        #    continue
        
        #check if it is not legal
        if expr.count("(") != expr.count(")"):
            failure_complete.append(i)
            continue
        
        #check if has legal op
        legal_op = True
        for op in ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "exp", "log", "sqrt"]:
            if op in expr and op+"(" not in expr:
                failure_legalOperator.append(i)
                legal_op = False
                break
        if legal_op == False:
            continue

        #check if has inequality
        inequality = False
        for op in [">", "<", "="]:
            if op in expr:
                failure_inequality.append(i)
                inequality = True
                break
        if inequality:
            continue                

        #check variables
        try:
            expr = expr.replace("arcsin", "asin")
            expr = expr.replace("arccos", "acos")
            expr = expr.replace("arctan", "atan")
            
            sp_expr = sp.parse_expr(expr, local_dict=generator.locals)
            flag = [s in generator.variable_list for s in sp_expr.free_symbols]
            if not all(flag) and "exp" not in expr:
                failure_variable.append(i)
                continue
            elif not all(flag):
                try:
                    # for v2, we do not allow failure exp
                    assert False
                    expr = replace_dimension_failure(expr, generator)
                    context[i] = expr
                except:
                    failure_variable.append(i)
                    continue

        #catch other exception during parse_expr
        except:
            failure_sympy.append(i)
            continue

        #check if it includes infinity
        #check if it is const, i.e.toooo simple 
        try:
            sp_num = sp.simplify(sp_expr).evalf()
        except:
            failure_sympy.append(i)
            continue
        if sp.zoo in sp.preorder_traversal(sp_num):
            failure_const.append(i)
            continue
        elif isinstance(sp_num, sp.core.numbers.Float) or\
           isinstance(sp_num, sp.core.numbers.Zero) or\
           isinstance(sp_num, sp.core.numbers.NaN) or\
           isinstance(sp_num, sp.core.numbers.ComplexInfinity) or\
           isinstance(sp_num, sp.core.numbers.Infinity) or\
           isinstance(sp_num, sp.core.numbers.NegativeInfinity):
            failure_const.append(i)
            continue

        #check dimension
        dimension_flag = True
        try:
            d, f = _cal_dimension(expr, generator)
            assert f
        except:
            dimension_flag = False
        
        if dimension_flag == False:
            replace_variables = False

            # for v2, we do not allow failure exp
            #if "exp" in expr:
            #    expr = replace_dimension_failure(expr, generator, op="exp")
            #    replace_variables = True

            if replace_variables:
                context[i] = expr

                try:
                    d, f = _cal_dimension(expr, generator)
                    assert f
                except:
                    failure_dimension.append(i)
            
            else:
                failure_dimension.append(i)

    return failure_length, failure_complete, failure_legalOperator, failure_inequality, failure_variable, failure_sympy, failure_const, failure_dimension

def second_test(context, generator):
    second_failure = []
    incomplete_units = []
    variables_num_failure = []
    pbar = tqdm.tqdm(total=len(context))
    for i, expr in enumerate(context):
        pbar.update(1)
        try:
            node, variables = generator.infix_to_node(expr)
        except:
            second_failure.append(i)
            continue

        try:
            dfs_stack = [node]
            while dfs_stack:
                current_node = dfs_stack.pop(0)
                assert all(current_node.unit == current_node.unit.astype(np.int32))
                dfs_stack.extend(current_node.children)
        except:
            incomplete_units.append(i)
        
        if len(variables) > 10:
            variables_num_failure.append(i)
    return second_failure, incomplete_units, variables_num_failure

def simplify(generator, context, filt, _simplify=True):
    pbar = tqdm.tqdm(total=len(context))
    new_context = []
    for i, expr in enumerate(context):
        pbar.update(1)
        if i not in filt:
            if _simplify:
                new_context.append(
                    str(sp.simplify(sp.parse_expr(expr,local_dict=generator.locals))) 
                )
            else:
                new_context.append(expr)
    pbar.close()
    return new_context


if __name__ == "__main__":

    #init exp
    parser = get_parser()
    params = parser.parse_args()
    env = build_env(params)
    generator = env.generator

    context0 = []
    with open("/home/linhw/yingjie/data/generated_results_v3.jsonl", "r") as fi:
        for line in fi:
            expr = json.loads(line)["pred"]
            context0.append(expr)
    with open("/home/linhw/yingjie/data/generated_results_v4.jsonl", "r") as fi:
        for line in fi:
            expr = json.loads(line)["pred"]
            context0.append(expr)
    
    #context0 = context0[:1000]


    print("total expressions to be preprocess:")
    print(len(context0))

    context = remove_duplica(context0, generator)
    print("remove expression duplica:")
    print(len(context0) - len(context))

    f1, f2, f3, f4, f5, f6, f7, f8 = preprocess(context, generator)

    print("failure num:")
    print(len(f1), len(f2), len(f3), len(f4), len(f5), len(f6), len(f7), len(f8))

    print("success num:")
    print(len(context)-len(f1+f2+f3+f4+f5+f6+f7+f8))

    lis0 = simplify(generator, context, f1+f2+f3+f4+f5+f6+f7+f8, _simplify=True)

    #second test
    lis = remove_duplica(lis0, generator)
    print("remove expression duplica:")
    print(len(lis0) - len(lis))

    f9, f10, f11 = second_test(lis, generator)
    
    print("second test failure num:")
    print(len(f9), len(f10), len(f11))

    print("success num:")
    print(len(lis)-len(f9)-len(f10)-len(f11))

    lis0 = simplify(generator, lis, f9+f10+f11, _simplify=False)

    with open("./data/preprocess_results_final.json", "w", encoding="utf-8") as fi:
        json.dump(lis0, fi)
    with open("./data/failure_expr_idx.json", "w", encoding="utf-8") as fi:
        json.dump([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11], fi)

