operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "abs": 1,
    "inv": 1,
    "sqrt": 1,
    "log": 1,
    "exp": 1,
    "sin": 1,
    "arcsin": 1,
    "cos": 1,
    "arccos": 1,
    "tan": 1,
    "arctan": 1,
    "pow2": 1,
    "pow3": 1,
    "pow4": 1,
    "pow5": 1,
    "neg": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
}

operators_extra = {"pow": 2}

math_constants = ["E", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}


binary_complex_dic = {
    "add": 1,
    "sub": 1,
    "mul": 1,
    "div": 1,
    "pow": 3,
}

unary_complex_dic = {
    "sin": 3,
    "cos": 3, 
    "tan": 3, 
    "exp": 3, 
    "log": 3,
    "sqrt": 3,
    "arcsin": 5,
    "arccos": 5,
    "arctan": 5,
    "neg": 1, 
    "inv": 1,
    "pow2": 3,
    "pow3": 3,
    "pow4": 3,
    "pow5": 3,
    "tanh": 5,
    "sinh": 5,
    "cosh": 5,
    "abs": 2,
}