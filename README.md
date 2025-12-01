# A Neural Symbolic Model for Space Physics

This repository contains code for the paper [A Neural Symbolic Model for Space Physics](https://www.nature.com/articles/s42256-025-01126-3).

If you encounter any problems with this program, please contact Jie Ying yingj24@mails.tsinghua.edu.cn.

![OverallFramework](figs/frameworks.png)

## Installation

Using conda and the environment.yml file:

```
conda env create --name PhyReg --file=environment.yml
conda activate PhyReg
```

## A Quick Start

A pre-trained model training on 6M synthetic formulas is avaliable from [Google Drive](https://drive.google.com/drive/folders/14M0Ed0gvSKmtuTOornfEoup8l48IfEUW).

After downoading and replace it with the empty `model.pt` file, you can play with `example.ipynb` as a demo example.

Other data which is necessary for training, evaluation and physics applications can be downloaded from [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), and should be placed in the `data` directory.

The whole data required to reproduce the experiments is also avaliable at [FigShare](https://doi.org/10.6084/m9.figshare.29615831.v1).

## Training

To train a new Physics Regressor model on your own, use the following command with additional arguments (arg1,val1), (arg2,val2):

`python train.py --arg1 val1 --arg2 --val2`

We also includes a template for the training of our Physics Regressor model, using the following command:

`bash ./bash/train.sh`

The training process consists of 100 epochs, with each epoch containing 500 training steps. The training time for each epoch ranges from 10 to 30 minutes when using a single 80GB A100 GPU. Occasionally, you may encounter the "CUDA: Out of Memory" error due to insufficient memory. In such cases, you can reduce the `tokens_per_batch` parameter, which defines the maximum number of tokens per batch, and increase `max_epoch` or `n_steps_per_epoch` parameter to maintain the same amount of training data. However, this may lead to different training outcomes.

The most useful hyper-parameters are presented in `./bash/train.sh`, which is listed below, while the others related are specified in parsers.py, and environment hyper-parameters are in envs/environment.py.

- **`expr_train_data_path`**: The path to dataset for training. You can use our synthetic data in `data/exprs_train`, available at [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`expr_valid_data_path`**: The path to dataset for validation. You can use our synthetic data in `data/exprs_valid`, available at [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`sub_expr_train_path`**: The path to sub-formula dataset for training. You can use our synthetic data in `data/exprs_seperated_train`, available at [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`sub_expr_valid_path`**: The path to sub-formula dataset for validation. You can use our synthetic data in `data/exprs_seperated_valid`, available at [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), or use any specific data of your own.
- **`max_epoch`**: The maximum training epochs.
- **`n_steps_per_epoch`**: The maximum training steps for each epoch.
- **`max_len`**: The maximum number of datapoints for each formula.
- **`eval_size`**: The number of validation formulas after each training epoch.
- **`tokens_per_batch`**: The maximum token count per training batch.

## Evaluation

Using our pre-trained model to evaluate, run the following command to evaluate the performance on synthetic dataset or feynman dataset:

`bash ./bash/eval_synthetic.sh`

`bash ./bash/eval_feynman.sh`

If you want to use trained model on your own, just reload checkpoint by modifying parameter `reload_checkpoint` to the path of your training checkpoint.

The Divide-and-Conquer strategy require training of oracle model, which is a little time-consuming. If you want to skip this, you can also downloaded our small oracle model from [Google Drive](https://drive.google.com/drive/folders/1VfH7Rp25U_pE504uhEd7dhSyvsBSXHdo), which should be placed at `Oracle_model` directory.

Similarly, the most useful hyper-parameters for evaluation are presented in `./bash/eval_synthetic.sh`, which is listed below,

- **`eval_size`**: The numbers of formulas to evaluate.
- **`batch_size_eval`**: The numbers of formulas to evaluate per batch.
- **`filename`**: The path to save evaluation results.
- **`oraclename`**: The path to save oracle neural network model.
- **`max_len`**: The number of datapoints for each formula.
- **`reload_checkpoint`**: The path to reload model or checkpoint (default to `model.pt`).
- **`expr_test_data_path`**: The path to the evaluation dataset (default to `./data/exprs_test_ranked.json`).

## Applications

The `physical` directory contains 5 physics application including SSN prediction, equator plasma pressure prediction, solar differential rotation prediction, contribution function prediction, lunar tide effect prediction.

The data for each physics cases can be found from [Google Drive](https://drive.google.com/drive/folders/17rbDLb2ZBgK9DidJtb1nyBFmGtOokhYs), and should also be placed in the `./data` directory, as mentioned above.

There are five Jupyter notebooks in the `physical` directory, each corresponding to one of the five real-world physics Symbolic Regression cases in the paper. These applications depend on the code of PhysicsRegression. To generate the predicted formula using PhysicsRegression, you need to open the `PhysicsRegression` repository and run these notebooks.

## Lisence

This project is covered under the Apache 2.0 License.

## Citation

```
 @article{Ying_Lin_Yue_Chen_Xiao_Shi_Liang_Yau_Zhou_Ma_2025, 
  title={A neural symbolic model for space physics}, 
  volume={7}, 
  url={http://dx.doi.org/10.1038/s42256-025-01126-3}, 
  DOI={10.1038/s42256-025-01126-3}, number={10}, 
  journal={Nature Machine Intelligence}, 
  publisher={Springer Science and Business Media LLC}, 
  author={Ying, Jie and Lin, Haowei and Yue, Chao and Chen, Yajie and Xiao, Chao and Shi, Quanqi and Liang, Yitao and Yau, Shing-Tung and Zhou, Yuan and Ma, Jianzhu}, 
  year={2025}, 
  month=oct, 
  pages={1726–1741}, 
  language={en} 
}

```