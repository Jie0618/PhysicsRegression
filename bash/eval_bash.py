import sys, os
sys.path.append(os.getcwd())

import copy
import json
import torch
import numpy as np

from parsers import get_parser
import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import initialize_exp
from symbolicregression.model import build_modules
from symbolicregression.envs import build_env
from symbolicregression.trainer import Trainer
from evaluate import Evaluator

import warnings
warnings.filterwarnings("ignore", message="masked_fill_ received a mask with dtype torch.uint8")
warnings.filterwarnings("ignore", message="We've integrated functorch into PyTorch.*")
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated:*")

def init_eval(params):
    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
    env.rng = np.random.RandomState()
    modules = build_modules(env, params)

    if "model.pt" in params.reload_checkpoint and params.reload_checkpoint != "":
        trainer = Trainer(modules, env, params, path="model.pt", root="./")
    else:
        trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    return evaluator, logger

if __name__ == "__main__":
    parser = get_parser()

    #additional params
    parser.add_argument("--repeat_trials", type=int, default=0, help="repeat trials")
    parser.add_argument("--filename", type=str, default="", help="")
    parser.add_argument("--oraclename", type=str, default="", help="")
    parser.add_argument("--current_eval_pos", type=int, default=0, help="where eval are currently?")

    params = parser.parse_args()

    #params
    params.rescale = False
    params.generate_datapoints_distribution = "positive,single"
    params.max_trials = 1000
    params.sample_expr_num = -1
    params.max_number_bags = -1
 
    np.random.seed(2024+params.repeat_trials)

    if params.expr_data != "feynman" and\
          (params.expr_test_data_path is None or params.expr_test_data_path == ""):
        raise FileNotFoundError("Please follow README and place the testing dataset into `./data` directory.")

    evaluator, logger = init_eval(params)
    evaluator.set_env_copies(["test"])

    scores = evaluator.evaluate_oracle_mcts(
        "test",
        "functions",
        logger=logger,
        verbose=False,
        save_file=os.path.join(
                os.getcwd(), "eval_result", params.filename
            ),
        datatype="test",
        oracle_name=params.oraclename,
        verbose_res=True,
    )