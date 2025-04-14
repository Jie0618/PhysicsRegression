from symbolicregression.envs.node import Node

import numpy as np
import sympy as sp
import scipy
from scipy.optimize import minimize
import time
import tqdm

np.seterr(all='ignore')

BINARY_OP = {
    "add": 1,
    "sub": 1,
    "mul": 1, 
    #"div": 1,
    #"pow": 1,
}

UNARY_OP = {
    #"abs": 1,
    "inv": 1,
    "sqrt": 1,

    #"log": 1,
    "exp": 1,

    "sin": 1,
    #"arcsin": 1,
    "cos": 1,
    #"arccos": 1,
    "tan": 1,
    #"arctan": 1,

    "pow2": 1,
    "pow3": 1,

    "neg": 1,
}

SKELETON_OP = {
    "skeleton": 1
}

SETTING_OP = {
    "empty": 1
}

class Individual():
    def __init__(self, node, xy, loss=None, complexity=None, len=None):
        self.node = node
        self.xy = xy

        if loss is None:
            self.update()
        else:
            self.loss = loss
            self.complexity = complexity
            self.len = len


    def __len__(self):
        return self.len
    
    def update(self):
        self._update_loss()
        self._update_len()
        self._update_complexity()

    def _update_loss(self):
        (x, y) = self.xy
        y_pred = self.node.val(x)
        
        if any(np.isnan(y_pred)):
            self.loss = np.inf
        try:
            self.loss =  np.sqrt(np.mean(np.square(y_pred.reshape((-1)) - y.reshape((-1)))))
        except:
            self.loss = np.inf
        self.loss = np.mean(np.square(y_pred.reshape((-1)) - y.reshape((-1))))

    def _update_len(self):
        self.len = len(self.node)

    def _update_complexity(self):
        self.complexity = len(self)

        

    def copy(self):
        return Individual(self.node.copy(), self.xy, self.loss, self.complexity, self.len)

    def __repr___(self):
        return str(self.node)
    
    def __str__(self):
        return str(self.node)


class Population():
    def __init__(self, population):
        self.population = population

    def replace_oldest(self, individual):
        self.population.pop(0)
        self.population.append(individual)
    
    def pop_and_add(self, idx, individual):
        self.population.pop(idx)
        self.population.append(individual)

    def __len__(self):
        return len(self.population)
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < len(self.population):
            result = self.population[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration
    

class BestFrontier():
    def __init__(self):
        self.frontier = {}

    def add(self, individual, complexity):
        if complexity not in self.frontier and not np.isnan(individual.loss):
            self.frontier[complexity] = individual.copy()
        elif complexity in self.frontier and individual.loss < self.frontier[complexity].loss:
            self.frontier[complexity] = individual.copy()

    def union(self, frontier):
        assert isinstance(frontier, BestFrontier)
        for complexity, individual in frontier.frontier.items():
            self.add(individual, complexity)
    
    def random_choose(self):
        idx = np.random.choice(list(self.frontier.keys()))
        return self.frontier[idx]
    
    def __len__(self):
        return len(self.frontier)
    
    def __repr__(self):
        dic = sorted(self.frontier.items(), key=lambda x:x[0])
        for k,v in dic:
            print("complexity:{} --> ".format(k), end="")
            print(v, end="")
            print(" --> loss={}".format(v.loss))
        return ""
    
    def best(self):
        """Return the best node across all complexity"""
        for k,v in self.frontier.items():
            if abs(v.loss) < 1e-6:
                v.loss = 0
        dic = sorted(self.frontier.items(), key=lambda x:(x[1].loss, x[0]))
        return dic[0][1].node



class TimeRecorder():
    def __init__(self):
        self.timerecorder1 = {
            "evolve": 0,
            "simplify": 0,
            "optimize": 0,
            "migration": 0,
        }

        self.timerecorder2 = {}
    
    def record(self, item, time):
        if item in self.timerecorder1:
            self.timerecorder1[item] += time
        elif item in self.timerecorder2:
            self.timerecorder2[item] += time
        else:
            self.timerecorder2[item] = time

    def write_in(self, path):
        s = "time recorder:\n\n"

        for k,v in self.timerecorder1.items():
            s += "{}:{:4f}\n".format(k,v)

        s += "\n"

        for k,v in self.timerecorder2.items():
            s += "{}:{:4f}\n".format(k,v)

        with open(path, "w") as fi:
            fi.write(s)
    
class DataRecorder():
    def __init__(self):
        self.datarecorder = []

    def new_epoch(self):
        self.datarecorder.append([])

    def new_population(self, population):
        self.datarecorder[-1].append({
            "populations":{
                "original": [(str(individual.node), individual.loss) for individual in population]
            },
            "mutations":[],
            "simplifys":[],
            "optimizes":[],
        })

    def record_population(self, population, name):
        self.datarecorder[-1][-1]["populations"][name] = [
                (str(individual.node), individual.loss) for individual in population
            ]
    
    def record_mutation(self, mutate_op, old, new):
        self.datarecorder[-1][-1]["mutations"].append([
            mutate_op, str(old.node), old.loss, str(new.node), new.loss
        ])
    
    def record_cross(self, old1, old2, new1, new2):
        self.datarecorder[-1][-1]["mutations"].append([
            "crossover", str(old1), old1.loss, str(old2), old2.loss, str(new1), new1.loss, str(new2), new2.loss
        ])
    
    def record_simplify(self, individual, state):
        if state == "before":
            self.info = str(individual.node)
        elif state == "after":
            self.datarecorder[-1][-1]["simplifys"].append([
                self.info, str(individual.node)
            ])
        else:
            print("unknown state:{}".format(state))

    def record_optimize(self, individual, state):
        if state == "before":
            self.info = [str(individual.node), individual.loss]
        elif state == "after":
            self.datarecorder[-1][-1]["optimizes"].append([
                self.info[0], self.info[1], str(individual.node), individual.loss
            ])
        else:
            print("unknown state:{}".format(state))
    

    def write_in(self, path):
        i = len(self.datarecorder) - 1
        j = len(self.datarecorder[-1]) - 1

        if ".txt" in path:
            path = path[:-4] + f"{i}_{j}.txt"
        else:
            path = path + f"{i}_{j}.txt"

        s = f"epoch:{i}, population:{j}\n\n"

        for name, population in self.datarecorder[i][j]["populations"].items():
            s += f"\n{name}:\n\n"
            for (node, loss) in population:
                s += "node:{}, loss:{:.4f}\n".format(node, loss)

        s += "\nmutation operation:\n\n"
        for k in self.datarecorder[i][j]["mutations"]:
            if k[0] == "crossover":
                s += "crossr, node1:{}\n        node2:{}\n         --> {}\n         --> {}\n          loss:{:.4f},{:.4f} --> {:.4f},{:.4f}\n\n".format(k[1], k[3], k[5], k[7], k[2], k[4], k[6], k[8])
            else:
                s += "{}, node:{}\n         --> {}\n        loss:{:.4f} --> {:.4f}\n\n".format(k[0], k[1], k[3], k[2], k[4])

        s += "\nsimplify:\n\n"
        for k in self.datarecorder[i][j]["simplifys"]:
            s += f"simplify, node:{k[0]}\n           --> {k[1]}\n\n"

        s += "\noptimize:\n\n"
        for k in self.datarecorder[i][j]["optimizes"]:
            s += "optimize, node:{}\n           --> {}\n          loss:{:.4f} --> {:.4f}\n\n".format(k[0], k[2], k[1], k[3])

        with open(path, "w") as fi:
            fi.write(s)

def bfgs_gp(node, xy, oracle, include_const=True):
    """
    const optimize for genetic programming

    first take out all the const in the given node, 

    then do the optimize steps, 

    and finally replace the const back into the node.

    if there are no or more than 6 const, then we refuse to do optimize

    Parameters:
    -----------
    node : Node
        the node form of expression to optimize
    
    xy : tuple
        input-output data pairs, of shape ((n_data, n_features), (n_data, 1))

    Returns:
    --------
    node : Node
        the node form of expression after optimize
    """

    safe_types = ["id", "safe", "neg", "inv", "safe-neg", "safe-inv"]
    if include_const and len(node) < node.params.max_complexity - 5 and np.random.random() < 0.6:
        safe_types.append("linear")
        use_const = True
    else:
        use_const = False

    refined_node = oracle.safely_refine(xy[0], xy[1], node, "id", safe_types)

    if use_const and len(refined_node) == len(node) + 5:
        if abs(float(node.children[0].children[1].value) - 1) < 1e-6:
            node.children[0] = node.children[0].children[0]
        if abs(float(node.children[1].value)) < 1e-6:
            node = node.children[0]

    return refined_node

class TreeGenerator(): 
    def __init__(self,
                 params,
                 N_max=200,
                 k_max=10,
                 precision=4,
                 max_number=10**100,
                 ):
        """
        `params`: control some fixed params in the expression tree.

        `b_max`: the maximum number of binary operator, default to be 5+D, deciding during generating

        `N_max`: the maximum number of input-output pairs

        `k_max`: the maximum number of the input cluster

        `precision`: the precision of each coefficient

        `max_number`: the maximum bound for y value
        """
        self.params = params
        self.D_max = params.D_max
        self.u_max = params.u_max
        self.N_max = N_max
        self.k_max = k_max
        self.precision = precision
        self.max_number = max_number

        self.UNARY_OP = UNARY_OP
        self.BINARY_OP = BINARY_OP
        self.SKELETON_OP = SKELETON_OP
        self.SETTING_OP = SETTING_OP

        self.unary_op = list(self.UNARY_OP.keys())
        self.binary_op = list(self.BINARY_OP.keys())
        self.skeleton_op = list(self.SKELETON_OP.keys())
        self.setting_op = list(self.SETTING_OP.keys())

        self.variables = ["x_{}".format(i) for i in range(1, self.D_max+1)]

        self.D_binary = self.generate_dist_for_binary(5 + self.D_max)

        self.unary_op_probability = np.array(list(self.UNARY_OP.values()))/sum(self.UNARY_OP.values())
        self.binary_op_probability = np.array(list(self.BINARY_OP.values()))/sum(self.BINARY_OP.values())


    def generate_dist_for_binary(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + D(n - 1, e + 1)
        TODO:finish unary-binary tree construct
        """
        D = np.zeros((2*max_ops+1, 2*max_ops+1)).astype(np.int64)
        for e in range(1, 2*max_ops+1):
            D[e][0] = 1
        for n in range(1, 2*max_ops+1):
            for e in range(1, 2*max_ops-n+1):
                D[e][n] = D[e-1][n] + D[e+1][n-1]
        return D
    
    def _generate_binary_tree(self, b, rng=np.random):
        """
        genrate a binary operator tree from "DEEP LEARNING FOR SYMBOLIC MATHEMATICS"

        `b`: given the number of operators

        `rng`: random number generator, default to np.random
        """
        e = 1               #e is the number of empty node
        n = b               #n is the number of op yet to be generate

        root = Node(0, self.params)
        nodes = [root]
        ptr = -1

        while n > 0:
            weights = self.D_binary[:e, n]/np.sum(self.D_binary[:e, n]) if e != 1 else [1]
            pos = rng.choice(range(e), p=weights)

            op = rng.choice(self.binary_op, p=self.binary_op_probability)

            ptr += pos +1
            nodes[ptr].update_value(op)
            left = Node(0, self.params)
            right = Node(0, self.params)
            nodes[ptr].push_children(left)
            nodes[ptr].push_children(right)
            nodes += [left,right]
            
            e = e - pos +1
            n -= 1
        
        return root, nodes
    
    def _add_variables(self, nodes, D, all_variables=True, rng=np.random):
        """
        add variables for self.generate_tree

        `nodes`: 
            all the nodes of the given expression

        `D`: 
            numbers of variable

        `all_variables`: 
            decide whether use all the variables,
            True if it is for generater random tree, 
            False if it is for generate tree with given complexity

        `rng`: 
            random number generator, default to np.random
        """
        leafs = []
        for node in nodes:
            if node.children == []:
                leafs.append(node)

        # assign all the variables
        if all_variables:   
            for i in range(1, D+1):
                pos = rng.randint(0, len(leafs))
                leafs[pos].update_value("x_{}".format(i))
                leafs.pop(pos)

        #assign the rest randomly
        for node in nodes:
            if node in leafs:
                node.update_value("x_{}".format(rng.randint(1, D+1)))

    def _add_unary_operators(self, prefix, u, rng=np.random):
        """
        add unary operators for self.generate_tree

        `prefix`: the prefix form of the given expression

        `u`: numbers of the unary operator

        `rng`: random number generator, default to np.random
        """
        prefix = prefix.split(',')
        for _ in range(u):
            pos = rng.randint(0, len(prefix)+1)
            op = rng.choice(self.unary_op, p=self.unary_op_probability)
            prefix.insert(pos, op)
        return ",".join(prefix)
    
    def _generate_float(self, rng=np.random):
        """
        generate random float for self._add_linear_transformations

        `rng`: random number generator, default to np.random
        """
        sign = rng.choice([-1, 1])
        mantissa = float(rng.choice(range(1, 10 ** self.precision)))
        exponent = rng.randint(-2-self.precision,2-self.precision+1)
        constant = sign * (mantissa * 10 ** exponent)
        return str(constant)
    
    def _add_linear_transformations(self, prefix, target, rng=np.random):
        """
        add linear transformation for target in self.generate_tree

        `prefix`: the prefix form of the given expression

        `target`: List, containing everything which need to take linear transformation

        `rng`: random number generator, default to np.random
        """
        prefix = prefix.split(',')

        offset = 0
        for i in range(len(prefix)):
            if prefix[i + offset] in target:
                a = self._generate_float(rng)
                b = self._generate_float(rng)
                prefix = prefix[:i + offset] + ["+", a, "*", b] + prefix[i+offset:]
                offset += 4
        return ",".join(prefix)

    def generate_tree_given_dimension_and_complexity(self, num_variables=None, complexity=None, rng=np.random):
        if num_variables is None and complexity is None:
            return self.generate_tree(rng)
        elif num_variables is None:
            D = rng.randint(1, self.D_max + 1)
            b = rng.randint(0, (complexity-1)//2 + 1)
            u = complexity - 1 - 2 * b
        elif complexity is None:
            D = num_variables
            b_max = 5 + D
            b = rng.randint(D - 1, D + b_max + 1)
            u = rng.randint(0, self.u_max + 1)
        else:
            D = num_variables
            b = rng.randint(0, (complexity-1)//2 + 1)
            u = complexity - 1 - 2 * b

        #sample binary tree
        root, nodes = self._generate_binary_tree(b, rng)

        #assiging variable
        self._add_variables(nodes, D, False, rng)

        #changing into prefix form
        prefix = root.prefix()

        #sample unary operator
        prefix =self._add_unary_operators(prefix, u, rng)

        root, _ = self.prefix_to_node(prefix)
        return root
    
    def prefix_to_node(self, prefix):
        """
        changing a prefix form to node form, i.e. infix form

        return a tuple (node, _)

        `prefix`: prefix form for changing
        """
        if prefix == "":return
        prefix = prefix.split(',')
        op = prefix[0]
        
        if op[:2] == 'x_':#variable
            return Node(op, self.params), ','.join(prefix[1:])
        try:#coefficent
            _ = float(op)
            return Node(op, self.params), ','.join(prefix[1:])
        except:
            pass
        if op in self.setting_op or op in self.skeleton_op:
            return Node(op, self.params), ','.join(prefix[1:])

        if op in self.unary_op:
            node = Node(op, self.params)
            child_node, prefix = self.prefix_to_node(','.join(prefix[1:]))
            node.push_children(child_node)
            return node, prefix
        elif op in self.binary_op:
            node = Node(op, self.params)
            child_node, prefix = self.prefix_to_node(','.join(prefix[1:]))
            node.push_children(child_node)
            child_node, prefix = self.prefix_to_node(prefix[:])
            node.push_children(child_node)
            return node, prefix
        else:
            print("error, unseen op:" + op)
        return node, prefix[1:]


class GeneticProgramming():
    def __init__(self,
                 treegenerator,
                 params,
                 oracle,
                 constant_perturbation_scale=1,
                 minimum_perturbation=0.1,
                 prob_negative_mutate_const=0.5,
                 max_attemp=100,
                 frecency_factor=1,
                 ):
        
        self.generator = treegenerator
        self.params = params
        self.oracle = oracle

        self.tornament_size = params.tornament_size
        self.p_tornament = params.p_tornament
        self.constant_perturbation_scale = constant_perturbation_scale
        self.minimum_perturbation = minimum_perturbation
        self.prob_negative_mutate_const = prob_negative_mutate_const
        self.max_attemp = max_attemp
        self.alpha_temperature_scale = params.alpha_temperature_scale
        self.mutation_epoch = params.mutation_epoch
        self.p_crossover = params.p_crossover
        self.population_num = params.population_num
        self.population_size = params.population_size
        self.epochs = params.gp_epochs
        self.fraction_replaced = params.fraction_replaced
        self.fraction_replaced_hof = params.fraction_replaced_hof
        self.frecency_factor = frecency_factor
        self.max_complexity = params.max_complexity
        self.discount_factor = params.discount_factor
        self.eps = params.eps
        self.optimize_probability = params.optimize_probability
        self.parsimony = params.parsimony
        self.use_recorder = params.use_recorder

        self.mutate_weight = np.array([0.048, 0.47, 0.79, 5.1, 1.7, 0.0020, 0.00023, 0.21])

        self.UNARY_OP = UNARY_OP
        self.BINARY_OP = BINARY_OP
        self.SKELETON_OP = SKELETON_OP
        self.SETTING_OP = SETTING_OP

        self.unary_op = list(self.UNARY_OP.keys())
        self.binary_op = list(self.BINARY_OP.keys())
        self.skeleton_op = list(self.SKELETON_OP.keys())
        self.setting_op = list(self.SETTING_OP.keys())

        self.prob_unary_op = np.array(list(self.UNARY_OP.values()), dtype=np.float32)
        self.prob_binary_op = np.array(list(self.BINARY_OP.values()), dtype=np.float32)
        self.prob_all_op = np.concatenate((self.prob_unary_op, self.prob_binary_op), dtype=np.float32)
        self.prob_unary_op /= np.sum(self.prob_unary_op)
        self.prob_binary_op /= np.sum(self.prob_binary_op)
        self.prob_all_op /= np.sum(self.prob_all_op)
        
    
    def score(self, individual, baseline, frecency):
        C = individual.complexity
        l_pred = individual.loss
        l = l_pred / baseline + C * self.parsimony
        ll = l * np.exp(frecency[min(C, self.max_complexity)] * self.max_complexity)
        return ll


    def best_of_sample(self, sample, frecency, baseline):
        def sort_key(x):
            return 1e20 if np.isnan(x[1]) else x[1]
        
        lis = [[individual, self.score(individual, baseline, frecency)] for individual in sample]
        lis.sort(key=sort_key)

        prob = [self.p_tornament * (1 - self.p_tornament) ** k for k in range(len(lis))]
        prob = np.cumsum(prob)

        idxs = np.where(np.random.random() < prob)[0]

        if idxs.shape == (0,):
            idx = -1
        else:
            idx = idxs.min()

        return lis[idx][0]


    def tornament(self, population, frecency, baseline):
        choosing = np.random.choice(self.population_size, self.tornament_size)
        choosing = Population([population.population[idx] for idx in choosing])
        E = self.best_of_sample(choosing, frecency, baseline)
        return E


    def _random_node(self, node):
        if node.degree == 0:
            return node
        elif node.degree == 1:
            left = len(node.children[0])
            prob = np.random.randint(1 + left)
            if prob == 0:
                return node
            else:
                return self._random_node(node.children[0])
        elif node.degree == 2:
            left = len(node.children[0])
            right = len(node.children[1])
            prob = np.random.randint(1 + left + right)
            if prob == 0:
                return node
            elif prob < 1 + left:
                return self._random_node(node.children[0])
            else:
                return self._random_node(node.children[1])
        else:
            print("error!!!")
            print("unexpected node.value:", node.value)

    def _random_node_with_parent(self, node, parent=None, idx=-1):
        if node.degree == 0:
            return node, parent, idx
        elif node.degree == 1:
            left = len(node.children[0])
            prob = np.random.randint(1 + left)
            if prob == 0:
                return node, parent, idx
            else:
                return self._random_node_with_parent(node.children[0], node, 0)
        elif node.degree == 2:
            left = len(node.children[0])
            right = len(node.children[1])
            prob = np.random.randint(1 + left + right)
            if prob == 0:
                return node, parent, idx
            elif prob < 1 + left:
                return self._random_node_with_parent(node.children[0], node, 0)
            else:
                return self._random_node_with_parent(node.children[1], node, 1)
        else:
            print("error!!!")
            print("unexpected node.value:", node.value)
        

    def make_random_leaf(self, num_variables):
        if np.random.random() < 0.5:
            return Node(np.random.random(), self.params)
        else:
            return Node("x_{}".format(np.random.randint(num_variables)), self.params)

    def _count_const(self, node):
        if node.is_const:
            return 1
        elif node.degree == 1:
            return self._count_const(node.children[0])
        elif node.degree == 2:
            return self._count_const(node.children[0]) + self._count_const(node.children[1])
        return 0

    def _count_operator(self, node):
        if node.value in self.unary_op:
            return 1 + self._count_operator(node.children[0])
        elif node.value in self.binary_op:
            return 1 + self._count_operator(node.children[0]) + self._count_operator(node.children[1])
        else:
            return 0


    def mutate_const(self, individual, T):
        if self._count_const(individual.node) == 0:
            return False

        node = self._random_node(individual.node)
        while node.degree != 0 or not node.is_const:
            node = self._random_node(individual.node)

        maxChange = 1 + self.constant_perturbation_scale * T + self.minimum_perturbation
        factor = maxChange ** np.random.random()
        makeConstBigger = np.random.random() > 0.5
        
        if makeConstBigger:
            new_value = float(node.value) * factor
        else:
            new_value = float(node.value) / factor

        if np.random.random() > self.prob_negative_mutate_const:
            new_value = - new_value
        
        node.update_value(new_value)

        return True


    def mutate_operator(self, individual):
        if self._count_operator(individual.node) == 0:
            return False

        node = self._random_node(individual.node)
        while node.degree < 1:
            node = self._random_node(individual.node)

        if node.degree == 1:
            new_op = np.random.choice(self.unary_op, p=self.prob_unary_op)
            node.update_value(new_op)
        elif node.degree == 2:
            new_op = np.random.choice(self.binary_op, p=self.prob_binary_op)
            node.update_value(new_op)

        return True


    def append_random_op(self, individual, num_variables, UNARY_ONLY=False):
        node, parent, idx = self._random_node_with_parent(individual.node)
        while node.degree > 0:
            node, parent, idx = self._random_node_with_parent(individual.node)

        if UNARY_ONLY:
            new_op = np.random.choice(self.unary_op, p=self.prob_unary_op)
        else:
            new_op = np.random.choice(self.unary_op + self.binary_op, p=self.prob_all_op)
        new_node = Node(new_op, self.params)

        if new_op in self.unary_op:
            new_node.push_children(node)
      
        elif new_op in self.binary_op:
            leaf = self.make_random_leaf(num_variables)
            if np.random.random() < 0.5:
                new_node.push_children(leaf)
                new_node.push_children(node)
            else:
                new_node.push_children(node)
                new_node.push_children(leaf)

        if parent is None:
            individual.node = new_node
        else:
            parent.children[idx] = new_node

    def append_random_op_PySR(self, individual, num_variables):
        node, parent, idx = self._random_node_with_parent(individual.node)
        while node.degree > 0:
            node, parent, idx = self._random_node_with_parent(individual.node)

        new_op = np.random.choice(self.unary_op + self.binary_op, p=self.prob_all_op)
        new_node = Node(new_op, self.params)

        if new_op in self.unary_op:
            leaf = self.make_random_leaf(num_variables)
            new_node.push_children(leaf)
       
        elif new_op in self.binary_op:
            left = self.make_random_leaf(num_variables)
            right = self.make_random_leaf(num_variables)
            new_node.push_children(left)
            new_node.push_children(right)

        if parent is None:
            individual.node = new_node
        else:
            parent.children[idx] = new_node


    def prepend_random_op(self, individual, num_variables): 
        node = individual.node
        new_op = np.random.choice(self.unary_op + self.binary_op, p=self.prob_all_op)
        new_node = Node(new_op, self.params)
        if new_op in self.unary_op:
            new_node.push_children(node)
        elif new_op in self.binary_op:
            leaf = self.make_random_leaf(num_variables)
            if np.random.random() < 0.5:
                new_node.push_children(leaf)
                new_node.push_children(node)
            else:
                new_node.push_children(node)
                new_node.push_children(leaf)
        individual.node = new_node


    def insert_random_op(self, individual, num_variables):
        node, parent, idx = self._random_node_with_parent(individual.node)

        new_op = np.random.choice(self.unary_op + self.binary_op, p=self.prob_all_op)
        new_node = Node(new_op, self.params)

        if new_op in self.unary_op:
            new_node.push_children(node)
        elif new_op in self.binary_op:
            leaf = self.make_random_leaf(num_variables)
            if np.random.random() < 0.5:
                new_node.push_children(leaf)
                new_node.push_children(node)
            else:
                new_node.push_children(node)
                new_node.push_children(leaf)
        
        if parent is None:
            individual.node = new_node
        else:
            parent.children[idx] = new_node

        return True

    def delete_random_node_PySR(self, individual, num_variables):
        node, parent, idx = self._random_node_with_parent(individual.node)

        if node.degree <= 0:
            new_node = self.make_random_leaf(num_variables)
            if parent is None:
                individual.node = new_node
            else:
                parent.children[idx] = new_node
            #node.update_value(new_node.value)
        elif node.degree == 1:
            if parent is None:
                individual.node = node.children[0]
            else:
                parent.children[idx] = node.children[0]
        elif node.degree == 2:
            if np.random.random() < 0.5:
                if parent is None:
                    individual.node = node.children[0]
                else:
                    parent.children[idx] = node.children[0]
            else:
                if parent is None:
                    individual.node = node.children[1]
                else:
                    parent.children[idx] = node.children[1]


    def generate_random_tree_PySR(self, individual, length, num_variables):
        individual.node = self.make_random_leaf(num_variables)
        for _ in range(length):
            self.append_random_op(individual, num_variables)

    def generate_random_tree_fixed_size_PySR(self, individual, size, num_variables):
        individual.node = self.make_random_leaf(num_variables)
        cur_size = 1
        
        while cur_size < size:
            if cur_size == size - 1:
                self.append_random_op(individual, num_variables, UNARY_ONLY=True)
            else:
                self.append_random_op(individual, num_variables)
            cur_size = len(individual.node)


    def _simplify_tree(self, node, bottom_up=True):
        """
        simplify given node through four steps:
        
        firstly, deal with the bottom node

        secondly, doing numerical calculate, such as: sin(2)

        thirdly, combine operators, such as: const + (xxx + const)

        fourthly, omit some useless const, such as: xxx + 0

        Parameters
        ----------
        node : Node
            given node to simplify
        
        bottom_up : Boolean
            decide whether goes from bottom

        Returns
        -------
        node : Node
            simplified node
        """

        node.update_value(node.value)
        assert len(node.children) == node.degree, f"Error in node: {str(node)}"

        if node.degree == 1:
            #firstly, deal with the bottom node
            if bottom_up:
                node.children[0] = self._simplify_tree(node.children[0])
            
            #secondly, numerical calculate
            if node.children[0].degree == 0 and node.children[0].is_const and np.isfinite(float(node.children[0].value)):
                val = node.val(np.array([0]))                #np.array([0]) cause it don't need xy
                return Node(float(val[0]), self.params)      #float(val[0]) to make sure the value is float and avoid turning a ndarray

            #thirdly, combine operator 

            #case 1
            #exp(log(xxx)) --> xxx
            #log(exp(xxx)) --> xxx
            elif node.value == "exp" and node.children[0].value == "log":
                return node.children[0].children[0]
            elif node.value == "log" and node.children[0].value == "exp":
                return node.children[0].children[0]

            #case 2
            #sin(arcsin(xxx)) --> xxx
            #cos(arccos(xxx)) --> xxx
            #tan(arctan(xxx)) --> xxx
            elif node.value == "sin" and node.children[0].value == "arcsin":
                return node.children[0].children[0]
            elif node.value == "arcsin" and node.children[0].value == "sin":
                return node.children[0].children[0]
            elif node.value == "cos" and node.children[0].value == "arccos":
                return node.children[0].children[0]
            elif node.value == "arccos" and node.children[0].value == "cos":
                return node.children[0].children[0]
            elif node.value == "tan" and node.children[0].value == "arctan":
                return node.children[0].children[0]
            elif node.value == "arctan" and node.children[0].value == "tan":
                return node.children[0].children[0]
            
            #case 3
            #-(-xxx) --> xxx
            elif node.value == "neg" and node.children[0].value == "neg":
                return node.children[0].children[0]
            
            #case 4
            #inv(inv(xxx)) --> xxx
            elif node.value == "inv" and node.children[0].value == "inv":
                return node.children[0].children[0]
            
            #case 5
            #sin(-xxx) --> -sin(xxx)
            #cos(-xxx) --> cos(xxx)
            #tan(-xxx) --> -tan(xxx)
            elif node.value == "sin" and node.children[0].value == "neg":
                tmp_node = Node("sin", self.params, children=[node.children[0].children[0]])
                tmpp_node = Node("neg", self.params, children=[tmp_node])
                return self._simplify_tree(tmpp_node, bottom_up=False)
            elif node.value == "cos" and node.children[0].value == "neg":
                node.children = node.children[0].children
                return self._simplify_tree(node, bottom_up=False)
            elif node.value == "tan" and node.children[0].value == "neg":
                tmp_node = Node("tan", self.params, children=[node.children[0].children[0]])
                tmpp_node = Node("neg", self.params, children=[tmp_node])
                return self._simplify_tree(tmpp_node, bottom_up=False)

            
            #fourthly, check some useless const

        elif node.degree == 2:
            #firstly, deal with the bottom node
            if bottom_up:
                
                node.children[0] = self._simplify_tree(node.children[0])
                node.children[1] = self._simplify_tree(node.children[1])


            #secondly, numerical calculate
            if node.children[0].degree == 0 and node.children[0].is_const and np.isfinite(float(node.children[0].value)) \
                and node.children[1].degree == 0 and node.children[1].is_const and np.isfinite(float(node.children[1].value)):
                val = node.val(np.array([0]))                #np.array([0]) cause it don't need xy
                return Node(float(val[0]), self.params)      #float(val[0]) to make sure the value is float and avoid turning a ndarray


            #thirdly, combine operator 

            if node.value == "add":
                
                #consider negative operator
                #(- xxx1) + xxx2 --> - (xxx2 - xxx1)
                if node.children[0].value == "neg":
                    tmp_node = Node("sub", self.params, children=[node.children[1], node.children[0].children[0]])
                    return self._simplify_tree(tmp_node, bottom_up=False)
                #xxx1 + (- xxx2) --> xxx1 - xxx2
                elif node.children[1].value == "neg":
                    tmp_node = Node("sub", self.params, children=[node.children[0], node.children[1].children[0]])
                    return self._simplify_tree(tmp_node, bottom_up=False)
                

                op = None
                if node.children[0].value == "add" and node.children[1].is_const:
                    #(const + xxx) + const --> const + xxx
                    if node.children[0].children[0].is_const:
                        const = float(node.children[1].value) + float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "add"
                    #(xxx + const) + const --> const + xxx
                    elif node.children[0].children[1].is_const:
                        const = float(node.children[1].value) + float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "add"

                elif node.children[0].value == "sub" and node.children[1].is_const and len(node.children[0].children) == 2:
                    #(const - xxx) + const --> const - xxx
                    if node.children[0].children[0].is_const:
                        const = float(node.children[1].value) + float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "sub"
                    #(xxx - const) + const --> const + xxx
                    elif node.children[0].children[1].is_const:
                        const = float(node.children[1].value) - float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "add"

                elif node.children[1].value == "add" and node.children[0].is_const:
                    #const + (const + xxx) --> const + xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) + float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "add"
                    #const + (xxx + const) --> const + xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) + float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "add"

                elif node.children[1].value == "sub" and node.children[0].is_const and len(node.children[1].children) == 2:
                    #const + (const - xxx) --> const - xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) + float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "sub"
                    #const + (xxx - const) --> const + xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) - float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "add"
                
                if op is not None:
                    if abs(const) < self.eps:
                        return right
                    else:
                        root = Node(op, self.params)
                        root.push_children(Node(const, self.params))
                        root.push_children(right)
                        return root
            
            elif node.value == "sub":
            
                #consider negative operator
                #(- xxx1) - xxx2 --> - (xxx1 + xxx2)
                if node.children[0].value == "neg":
                    tmp_node = Node("add", self.params, children=[node.children[1], node.children[0].children[0]])
                    tmpp_node = Node("neg", self.params, children=[tmp_node])
                    return self._simplify_tree(tmpp_node, bottom_up=False)
                #xxx1 - (- xxx2) --> xxx1 + xxx2
                elif node.children[1].value == "neg":
                    tmp_node = Node("add", self.params, children=[node.children[0], node.children[1].children[0]])
                    return self._simplify_tree(tmp_node, bottom_up=False)

                op = None
                if node.children[0].value == "add" and node.children[1].is_const:
                    #(const + xxx) - const --> const + xxx
                    if node.children[0].children[0].is_const:
                        const = - float(node.children[1].value) + float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "add"
                    #(xxx + const) - const --> const + xxx
                    elif node.children[0].children[1].is_const:
                        const = - float(node.children[1].value) + float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "add"

                elif node.children[0].value == "sub" and node.children[1].is_const and len(node.children[0].children) == 2:
                    #(const - xxx) - const --> const - xxx
                    if node.children[0].children[0].is_const:
                        const = - float(node.children[1].value) + float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "sub"
                    #(xxx - const) - const --> const + xxx
                    elif node.children[0].children[1].is_const:
                        const = - float(node.children[1].value) - float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "add"

                elif node.children[1].value == "add" and node.children[0].is_const:
                    #const - (const + xxx) --> const - xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) - float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "sub"
                    #const - (xxx + const) --> const - xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) - float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "sub"

                elif node.children[1].value == "sub" and node.children[0].is_const and len(node.children[1].children) == 2:
                    #const - (const - xxx) --> const + xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) - float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "add"
                    #const - (xxx - const) --> const - xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) + float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "sub"
                
                if op is not None:
                    if abs(const) < self.eps and op == "add":
                        return right
                    elif abs(const) < self.eps and op == "sub":
                        return Node("neg", self.params, children=[right])
                    else:
                        root = Node(op, self.params)
                        root.push_children(Node(const, self.params))
                        root.push_children(right)
                        return root
                    

            if node.value == "mul":
                
                #consider negative operator
                #(- xxx1) * (- xxx2) --> xxx1 * xxx2
                if node.children[0].value == "neg" and node.children[1].value == "neg":
                    tmp_node = Node("mul", self.params, children=[node.children[0].children[0], node.children[1].children[0]])
                    return self._simplify_tree(tmp_node, bottom_up=False)
                #(- xxx1) * xxx2 --> - (xxx1 * xxx2)
                elif node.children[0].value == "neg":
                    tmp_node = Node("mul", self.params, children=[node.children[0].children[0], node.children[1]])
                    tmpp_node = Node("neg", self.params, children=[tmp_node])
                    return self._simplify_tree(tmpp_node, bottom_up=False)
                #xxx1 * (- xxx2) --> - (xxx1 * xxx2)
                elif node.children[1].value == "neg":
                    tmp_node = Node("mul", self.params, children=[node.children[0], node.children[1].children[0]])
                    tmpp_node = Node("neg", self.params, children=[tmp_node])
                    return self._simplify_tree(tmpp_node, bottom_up=False)


                op = None
                if node.children[0].value == "mul" and node.children[1].is_const:
                    #(const * xxx) * const --> const * xxx
                    if node.children[0].children[0].is_const:
                        const = float(node.children[1].value) * float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "mul"
                    #(xxx * const) * const --> const * xxx
                    elif node.children[0].children[1].is_const:
                        const = float(node.children[1].value) * float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "mul"
                elif node.children[0].value == "div" and node.children[1].is_const:
                    #(const / xxx) * const --> const / xxx
                    if node.children[0].children[0].is_const:
                        const = float(node.children[1].value) * float(node.children[0].children[0].value)
                        right = node.children[0].children[1]
                        op = "div"
                    #(xxx / const) * const --> const * xxx
                    elif node.children[0].children[1].is_const:
                        const = float(node.children[1].value) / float(node.children[0].children[1].value)
                        right = node.children[0].children[0]
                        op = "mul"

                elif node.children[1].value == "mul" and node.children[0].is_const:
                    #const * (const * xxx) --> const * xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) * float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "mul"
                    #const * (xxx * const) --> const * xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) * float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "mul"
                elif node.children[1].value == "div" and node.children[0].is_const:
                    #const * (const / xxx) --> const / xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) * float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "div"
                    #const * (xxx / const) --> const * xxx
                    elif node.children[1].children[1].is_const:
                        try:
                            const = float(node.children[0].value) / float(node.children[1].children[1].value)
                            right = node.children[1].children[0]
                            op = "mul"
                        except:
                            pass


                if op is not None:
                    if np.abs(const - 1) < self.eps:
                        return right
                    elif np.abs(const) < self.eps:
                        return Node(0, self.params)
                    else:
                        root = Node(op, self.params)
                        root.push_children(Node(const, self.params))
                        root.push_children(right)
                        return root
            
            if node.value == "div":

                #consider negative operator
                #(- xxx1) / (- xxx2) --> xxx1 / xxx2
                if node.children[0].value == "neg" and node.children[1].value == "neg":
                    tmp_node = Node("div", self.params, children=[node.children[0].children[0], node.children[1].children[0]])
                    return self._simplify_tree(tmp_node, bottom_up=False)
                #(- xxx1) / xxx2 --> - (xxx1 / xxx2)
                elif node.children[0].value == "neg":
                    tmp_node = Node("div", self.params, children=[node.children[0].children[0], node.children[1]])
                    tmpp_node = Node("neg", self.params, children=[tmp_node])
                    return self._simplify_tree(tmpp_node, bottom_up=False)
                #xxx1 / (- xxx2) --> - (xxx1 / xxx2)
                elif node.children[1].value == "neg":
                    tmp_node = Node("mul", self.params, children=[node.children[0], node.children[1].children[0]])
                    tmpp_node = Node("neg", self.params, children=[tmp_node])
                    return self._simplify_tree(tmpp_node, bottom_up=False)
                
                op = None
                if node.children[0].value == "mul" and node.children[1].is_const:
                    #(const * xxx) / const --> const * xxx
                    if node.children[0].children[0].is_const:
                        const = float(node.children[0].children[0].value) * float(node.children[1].value)
                        right = node.children[0].children[1]
                        op = "mul"
                    #(xxx * const) / const --> const * xxx
                    elif node.children[0].children[1].is_const:
                        const = float(node.children[0].children[1].value) * float(node.children[1].value)
                        right = node.children[0].children[0]
                        op = "mul"
                elif node.children[0].value == "div" and node.children[1].is_const:
                    #(const / xxx) / const --> const / xxx
                    if node.children[0].children[0].is_const:
                        try:
                            const = float(node.children[0].children[0].value) / float(node.children[1].value)
                            right = node.children[0].children[1]
                            op = "div"
                        except:
                            pass

                    #(xxx / const) / const --> const * xxx
                    elif node.children[0].children[1].is_const:
                        try:
                            const = 1 / (float(node.children[1].value) * float(node.children[0].children[1].value))
                            right = node.children[0].children[0]
                            op = "mul"
                        except:
                            pass

                elif node.children[1].value == "mul" and node.children[0].is_const:
                    #const / (const * xxx) --> const / xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) / float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "div"
                    #const / (xxx * const) --> const / xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) / float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "div"
                elif node.children[1].value == "div" and node.children[0].is_const:
                    #const / (const / xxx) --> const * xxx
                    if node.children[1].children[0].is_const:
                        const = float(node.children[0].value) / float(node.children[1].children[0].value)
                        right = node.children[1].children[1]
                        op = "mul"
                    #const / (xxx / const) --> const / xxx
                    elif node.children[1].children[1].is_const:
                        const = float(node.children[0].value) / float(node.children[1].children[1].value)
                        right = node.children[1].children[0]
                        op = "div"

                if op is not None:
                    if np.abs(const - 1) < self.eps and op == "mul":
                        return right
                    elif np.abs(const) < self.eps:
                        return Node(0, self.params)
                    else:
                        root = Node(op, self.params)
                        root.push_children(Node(const, self.params))
                        root.push_children(right)
                        return root
                

            # xxx / xxx --> 1
            if node.value == "div" and str(node.children[0]) == str(node.children[1]):
                return Node(1, self.params)


            # xxx - xxx ---> 0
            elif node.value == "sub" and str(node.children[0]) == str(node.children[1]):
                return Node(0, self.params)
            
            
            #fourthly, check some useless const

            #case: 0 + xxx or xxx + 0
            if node.value == "add":
                if node.children[0].is_const and abs(float(node.children[0].value)) < self.eps:
                    return node.children[1]
                elif node.children[1].is_const and abs(float(node.children[1].value)) < self.eps:
                    return node.children[0]

            #case: xxx - 0 or 0 - xxx
            elif node.value == "sub":
                if node.children[1].is_const and abs(float(node.children[1].value)) < self.eps:
                    return node.children[0]
                elif node.children[0].is_const and abs(float(node.children[0].value)) < self.eps:
                    return Node("neg", self.params, children=[node.children[1]])

            #case: 0 * xxx or xxx * 0
            #case: 1 * xxx or xxx * 1
            elif node.value == "mul":
                if (node.children[0].is_const and abs(float(node.children[0].value)) < self.eps) or (node.children[1].is_const and abs(float(node.children[1].value)) < self.eps):
                    return Node(0, self.params)
                elif node.children[0].is_const and abs(float(node.children[0].value) - 1) < self.eps:
                    return node.children[1]
                elif node.children[1].is_const and abs(float(node.children[1].value) - 1) < self.eps:
                    return node.children[0]
                    
            #case: 0 / xxx
            #case: xxx / 1
            elif node.value == "div":
                if node.children[0].is_const and abs(float(node.children[0].value)) < self.eps:
                    return Node(0, self.params)
                elif node.children[1].is_const and abs(float(node.children[1].value) - 1) < self.eps:
                    return node.children[0]

            #case: 1 ^ xxx
            #case: xxx ^ 0
            #case: xxx ^ 1
            elif node.value == "pow":
                if node.children[0].is_const and abs(float(node.children[0].value) - 1) < self.eps:
                    return Node(1, self.params)
                elif node.children[1].is_const and abs(float(node.children[1].value)) < self.eps:
                    return Node(1, self.params)
                elif node.children[1].is_const and abs(float(node.children[1].value) - 1) < self.eps:
                    return node.children[0]
                
        return node

    def simplify_tree(self, individual):
        individual.node = self._simplify_tree(individual.node)


    def generate_random_tree(self, individual, num_variables):
        node = self.generator.generate_tree_given_dimension_and_complexity(num_variables, 3)
        individual.node = node

    def _check_unary(self, node):
        if node.value in UNARY_OP and node.children[0].value in UNARY_OP and node.children[0].children[0].value in UNARY_OP:
            return False
        elif node.value in UNARY_OP:
            return self._check_unary(node.children[0])
        elif node.value in BINARY_OP:
            return self._check_unary(node.children[0]) and self._check_unary(node.children[1])
        return True


    def check_constrain(self, individual):
        individual._update_len()
        individual._update_complexity()
        if individual.complexity > self.max_complexity:
            return False
        return self._check_unary(individual.node)
    

    def mutation(self, E, T, num_variables, baseline, frecency):   
        attemp = 0
        satisfy_constrain = False

        while attemp < self.max_attemp and not satisfy_constrain:

            mutate_weight = self.mutate_weight.copy()
            #If equation too big, don't add new operators
            if len(E) >= self.max_complexity:
                mutate_weight[2] = 0
                mutate_weight[3] = 0
            mutate_weight /= np.sum(mutate_weight)
            
            i = np.random.choice(8, p=mutate_weight)

            t1 = time.time()

            new_E = E.copy()

            t2 = time.time()

            self.timerecorder.record("copy", t2-t1)

            if i == 0:
                #mutate const
                t1 = time.time()
                self.mutate_const(new_E, T)
                satisfy_constrain = True
                t2 = time.time()
                self.timerecorder.record("mutate const", t2-t1)
            elif i == 1:
                #mutate operator
                t1 = time.time()
                self.mutate_operator(new_E)
                t2 = time.time()
                self.timerecorder.record("mutate op", t2-t1)
            elif i == 2:
                #append/prepend node
                t1 = time.time()
                if np.random.random() < 0.5:
                    self.prepend_random_op(new_E, num_variables)
                else:
                    self.append_random_op_PySR(new_E, num_variables)
                t2 = time.time()
                self.timerecorder.record("append prepend", t2-t1)
            elif i == 3:
                #insert node
                t1 = time.time()
                self.insert_random_op(new_E, num_variables)
                t2 = time.time()
                self.timerecorder.record("insert node", t2-t1)
            elif i == 4:
                #delete node
                t1 = time.time()
                self.delete_random_node_PySR(new_E, num_variables)
                satisfy_constrain = True
                t2 = time.time()
                self.timerecorder.record("delete node", t2-t1)
            elif i == 5:
                #simplify
                t1 = time.time()
                self.simplify_tree(new_E)
                satisfy_constrain = True
                t2 = time.time()
                self.timerecorder.record("simplify", t2-t1)
            elif i == 6:
                #random
                #tree_size_to_generate = np.random.randint(1, self.max_complexity)
                t1 = time.time()
                tree_size_to_generate = 3
                self.generate_random_tree_fixed_size_PySR(new_E, tree_size_to_generate, num_variables)
                t2 = time.time()
                self.timerecorder.record("ramdon gen", t2-t1)
            elif i == 7:
                #do nothing
                return E, "passss"

            #TODO:check constrain
            t1 = time.time()
            satisfy_constrain = satisfy_constrain or self.check_constrain(new_E)
            attemp += 1
            t2 = time.time()
            self.timerecorder.record("check_constrain", t2-t1)

        t1 = time.time()
        new_E.update()
        t2 = time.time()
        self.timerecorder.record("cal loss", t2-t1)

        ops = ["mut_ct", "mut_op", "append", "insert", "delete", "simpfy", "random", "passss"]

        if not satisfy_constrain:
            return None, None
        
        old_score = self.score(E, baseline, frecency)
        new_score = self.score(new_E, baseline, frecency)

        q_anneal = np.exp(-(new_score - old_score) / (self.alpha_temperature_scale * T))
        #q_parsimony = min(E.complexity / new_E.complexity, 1)
        #q_parsimony = E.complexity / new_E.complexity
        q_parsimony = frecency[
            min(E.complexity, self.params.max_complexity)
        ] / frecency[
            min(new_E.complexity, self.params.max_complexity)
        ]

        if np.isnan(E.loss) or np.random.random() < q_anneal * q_parsimony:
            return new_E, ops[i]
        return None, None


    def _cross_over(self, E1, E2):
        node1, parent1, idx1 = self._random_node_with_parent(E1.node)
        node2, parent2, idx2 = self._random_node_with_parent(E2.node)

        if parent1 is None and parent2 is None:
            E1.node = node2
            E2.node = node1
        elif parent1 is None:
            E1.node = node2
            parent2.children[idx2] = node1
        elif parent2 is None:
            E2.node = node1
            parent1.children[idx1] = node2
        else:
            parent1.children[idx1] = node2
            parent2.children[idx2] = node1

    def cross_over(self, E1, E2):
        attemp = 0
        satisfy_constrain = False
        while attemp < self.max_attemp and not satisfy_constrain:
            new_E1 = E1.copy()
            new_E2 = E2.copy()
            self._cross_over(new_E1, new_E2)
            satisfy_constrain = self.check_constrain(new_E1) and self.check_constrain(new_E2)
            attemp += 1
        if not satisfy_constrain:
            return None, None

        new_E1.update()
        new_E2.update()

        return new_E1, new_E2


    def evolve(self, population, num_variables, frecency, baseline):
        for k in range(self.mutation_epoch):
            if np.random.random() > self.p_crossover:
                #mutate
                individual = self.tornament(population, frecency, baseline)
                T = 1 - k / self.mutation_epoch
                t1 = time.time()
                new_individual, op = self.mutation(individual, T, num_variables, baseline, frecency)
                t2 = time.time()
                self.timerecorder.record("mutate", t2-t1)

                #replace the oldest
                if new_individual is not None:
                    population.replace_oldest(new_individual)
                    self.datarecorder.record_mutation(op, individual, new_individual)

            else:
                #crossover
                individual1 = self.tornament(population, frecency, baseline)
                individual2 = self.tornament(population, frecency, baseline)
                t1 = time.time()
                new_individual1, new_individual2 = self.cross_over(individual1, individual2)
                t2 = time.time()
                self.timerecorder.record("crossover", t2-t1)
            
                #replace the oldest
                if new_individual1 is not None:
                    population.replace_oldest(new_individual1)
                    population.replace_oldest(new_individual2)
                    self.datarecorder.record_cross(individual1, individual2, new_individual1, new_individual2)


    def optimize_and_simplify_population(self, population, xy):
        rand = np.random.random(size=(len(population)))
        for i, individual in enumerate(population):

            self.datarecorder.record_simplify(individual, "before")
            t1 = time.time()

            self.simplify_tree(individual)

            t2 = time.time()

            self.timerecorder.record("simplify", t2-t1)
            self.datarecorder.record_simplify(individual, "after")

            individual._update_len()
            individual._update_complexity()

            t3 = time.time()

            #optimize with probability
            if rand[i] < self.optimize_probability:
                self.datarecorder.record_optimize(individual, "before")
                individual.node = bfgs_gp(individual.node, xy, self.oracle)
                individual._update_loss()
                self.datarecorder.record_optimize(individual, "after")

                t4 = time.time()

                self.timerecorder.record("optimize", t4-t3)

                ###TODO:fix this!!!
                self.datarecorder.record_simplify(individual, "before")
                t1 = time.time()

                self.simplify_tree(individual)

                t2 = time.time()

                self.timerecorder.record("simplify", t2-t1)
                self.datarecorder.record_simplify(individual, "after")

                individual._update_len()
                individual._update_complexity()

    def update_frecency(self, frecency):
        sumi = np.sum(frecency)
        window_size = 100000
        different_size = sumi - window_size
        num_loop = 1
        while different_size > 0:
            idx_to_subtract = np.where(frecency > 1)
            num_remaining = idx_to_subtract.shape[0]
            amount_to_subtract = np.min(
                different_size / num_remaining,
                min(frecency[idx_to_subtract] - 1)
            )
            frecency[idx_to_subtract] -= amount_to_subtract
            total_amount_to_subtract = amount_to_subtract * num_remaining
            different_size -= total_amount_to_subtract
            num_loop += 1
            if num_loop > 1000:
                break

    def migration(self, population, best_of_population, best_of_all, index):
        t1 = time.time()

        j = 0; end = len(population) - 1
        while j <= end:

            #migration from the best frontier of other populations
            if np.random.random() < self.fraction_replaced:
                p = np.array([len(frontier) for frontier in best_of_population], dtype=np.float32)
                p[index] = 0
                if np.sum(p) == 0:break
                p /= np.sum(p)
                population_idx = np.random.choice(np.arange(self.population_num), p=p)
                E = best_of_population[population_idx].random_choose().copy()
                population.pop_and_add(j, E)
                j -= 1
                end -= 1

            #migration from the best frontier of all 
            if np.random.random() < self.fraction_replaced_hof:
                E = best_of_all.random_choose().copy()
                population.pop_and_add(j, E)
                j -= 1
                end -= 1

            j += 1

        t2 = time.time()
        self.timerecorder.record("migration", t2-t1)
        

    def run(self, env, num_variables, xy, exprs=None, verbose=1):

        populations = self.create_population(env, num_variables, xy, exprs)

        (x, y) = xy

        baseline = np.zeros_like(y) - y
        baseline = np.mean(np.square(baseline))

        #restore complexity-frequency
        frecency = np.ones(shape=(self.max_complexity + 1))

        #restore frontier of each population / all population
        best_of_all = BestFrontier()
        best_of_population = [BestFrontier() for _ in range(self.population_num)]

        self.timerecorder = TimeRecorder()
        self.datarecorder = DataRecorder()

        for epoch in range(self.epochs):
            
            if verbose:
                print("epoch:{}...".format(epoch+1))
                print()

            self.datarecorder.new_epoch()
            
            for i in range(self.population_num):

                population = populations[i]

                self.datarecorder.new_population(population)

                #aging
                for individual in population:
                    #individual.age += 1
                    #frecency[len(individual)] += 1
                    pass

                t1 = time.time()

                self.evolve(population, num_variables, frecency / sum(frecency), baseline)

                t2 = time.time()

                #self.timerecorder.record("evolve", t2-t1)
                #self.datarecorder.record_population(population, "after evolve")

                #simplify and optimize
                self.optimize_and_simplify_population(population, xy)

                #self.datarecorder.record_population(population, "after simplify")

                #record the best in this population at each complexity
                best_of_temp = best_of_population[i]
                for individual in population:
                    complexity = individual.complexity
                    best_of_temp.add(individual, complexity)
                best_of_population[i] = best_of_temp
                
                #record the best of all the populations
                best_of_all.union(best_of_temp)

                #migration
                self.migration(population, best_of_population, best_of_all, i)

                #self.datarecorder.record_population(population, "after migration")

                #self.datarecorder.write_in("./recorder/datarecorder.txt")

                #aging
                for individual in population:
                    frecency[min(len(individual), self.params.max_complexity)] += 1

                self.update_frecency(frecency)
            
            early_stop = False
            for k, v in best_of_all.frontier.items():
                if np.abs(v.loss) < self.eps:
                    early_stop = True
                    break
            if early_stop:
                break
            
            if verbose:
                print(best_of_all)

        #final optimize for frontier
        for k, v in best_of_all.frontier.items():
            v.node = bfgs_gp(v.node, xy, self.oracle, include_const=False)

        #self.timerecorder.write_in("./recorder/timerecorder.txt")
        
        
        return best_of_all


    def create_population(self, env, num_variables, xy, exprs=None, random_expr_rate=0.6):

        try:
            assert num_variables == xy[0].shape[1]
        except:
            print(exprs, xy[0].shape, num_variables)
            raise False

        if exprs is None:
            
            populations = []
            for i in range(self.population_num):
                population = []
                for j in range(self.population_size):
                    individual = Individual(Node(0, self.params), xy)
                    self.generate_random_tree_fixed_size_PySR(individual, 3, num_variables)
                    individual.update()

                    population.append(individual)
                population = Population(population)
                populations.append(population)
        
        else:

            nodes = []
            for i in range(len(exprs)):

                try:
                    node, variables = env.generator.infix_to_node(exprs[i], allow_pow=True, label_units=False, sp_parse=False)
                except:
                    print(exprs[i])
                    raise False
                nodes.append(node)

            num_exprs_per_populations = int((len(exprs)-1) / self.population_num) + 1

            populations = []
            for i in range(self.population_num):
                population = []
                cnt = 0
                for j in range(self.population_size):
                    if np.random.random() < random_expr_rate or num_exprs_per_populations * i >= len(exprs):
                        individual = Individual(Node(0, self.params), xy)
                        self.generate_random_tree_fixed_size_PySR(individual, 3, num_variables)
                        individual.update()
                    else:
                        node = nodes[num_exprs_per_populations * i + cnt]
                        cnt = (cnt + 1) % (min(
                                num_exprs_per_populations, len(nodes) - num_exprs_per_populations * i
                            ))
                        individual = Individual(node, xy)
                        individual.update()

                    population.append(individual)
                population = Population(population)
                populations.append(population)
        
        return populations

