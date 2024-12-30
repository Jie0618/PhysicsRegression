# A Neural Symbolic Model for Space Physics

This repository contains code for the paper [A Neural Symbolic Model for Space Physics](XXX).

## Install

Using conda and the environment.yml file:

```
conda env create --name PhyReg --file=environment.yml
conda activate PhyReg
```

## A Quick Start

A pre-trained model training on 6M synthetic formulas is avaliable from [link](https://drive.google.com/drive/folders/14M0Ed0gvSKmtuTOornfEoup8l48IfEUW).

After downoading and replace it with the empty `model.pt` file, you can play with `example.ipynb` as a demo example.

Other data which is necessary for training or evaluation can be downloaded from [link](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), and should be placed in the `data` directory.

## Training

To train a new Physics Regressor model on your own, use the following command with additional arguments (arg1,val1), (arg2,val2):

`python train.py --arg1 val1 --arg2 --val2`

We also includes a template for the training of our Physics Regressor model, using the following command:

`bash ./bash/train.sh`

The most useful hyper-parameters are presented in `./bash/train.sh`, which is listed below, while the others related are specified in parsers.py, and environment hyper-parameters are in envs/environment.py.

- **`expr_train_data_path`**: The path to dataset for training. You can use our synthetic data in `data/exprs_train`, avaiable at [link](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`expr_valid_data_path`**: The path to dataset for validation. You can use our synthetic data in `data/exprs_valid`, avaiable at [link](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`sub_expr_train_path`**: The path to sub-formula dataset for training. You can use our synthetic data in `data/exprs_seperated_train`, avaiable at [link](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`sub_expr_valid_path`**: The path to sub-formula dataset for validation. You can use our synthetic data in `data/exprs_seperated_valid`, avaiable at [link](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`max_epoch`**: The maximum training epochs.
- **`n_steps_per_epoch`**: The maximum training steps for each epoch.
- **`max_len`**: The maximum number of datapoints for each formula.
- **`eval_size`**: The number of validation formulas after each training epoch.

## Evaluation

Using our pre-trained model to evaluate, run the following command to evaluate the performance on synthetic dataset or feynman dataset:

`bash ./bash/eval_synthetic.sh`

`bash ./bash/eval_feynman.sh`

If you want to use trained model on your own, just reload checkpoint by modifying parameter `reload_checkpoint` to the path of your training checkpoint.

The Divide-and-Conquer strategy require training of oracle model, which is a little time-consuming. If you want to skip this, you can also downloaded our small oracle model from [link](https://drive.google.com/drive/folders/1VfH7Rp25U_pE504uhEd7dhSyvsBSXHdo), which should be placed at `Oracle_model` dictionary.

Similarly, the most useful hyper-parameters for evaluation are presented in `./bash/eval_synthetic.sh`, which is listed below,

- **`eval_size`**: The numbers of formulas to evaluate.
- **`batch_size_eval`**: The numbers of formulas to evaluate per batch.
- **`filename`**: The path to save evaluation results.
- **`oraclename`**: The path to save oracle neural network model.
- **`max_len`**: The number of datapoints for each formula.
- **`reload_checkpoint`**: The path to reload model or checkpoint (default to `model.pt`).

## Applications

The `physical` distionary contains 5 physics application including SSN prediction, equator plasma pressure prediction, solar differential rotation prediction, contribution function prediction, lunar tide effect prediction.

The data for each physics cases can be found from [link](https://drive.google.com/drive/folders/1mS_BA-T7xupP3KgiMQ_I6mVfheXf7H2X?usp=share_link), and should be placed in the `physical/data` directory in each physics cases.
