# Deep Clustering With Consensus Representations (DECCS) - ICDM 2022

The paper is available online here: [arxiv](https://arxiv.org/abs/2210.07063), icdm (as soon as proceedings are online)

This is the accompanying code to reproduce the main results of our DECCS algorithm published at ICDM 2022.  

The code has been implemented in Python 3.9.7 and tested on Ubuntu 18.04.2 LTS.
We implemented the autoencoders and DECCS in Pytorch 1.10.1 with cuda support.

## Tutorial - Notebook

In [synthetic-example.ipynb](synthetic-example.ipynb) we provide a step by step jupyter notebook of how DECCS is applied to the SYNTH data set described in the paper. It has many comments and plots. We hope it helps to understand the main steps of the algorithm.


## Installation

You can install the environment via:
conda create --name pytorch110 --file environment.yml

## Data Sets and pretrained models

You can download all data sets and models from zenodo: https://doi.org/10.5281/zenodo.7330360

Unzip the data.zip files in the data directory and the pretrained autoencoders per data set in the results directory. The small UCI data sets can already be found in the data directory and the pretrained models for the synthetic data are already in the results directory. The image data sets are automatically downloaded via PyTorch.

## Pretrain autoencoders yourself

If you want to pretrain the autoencoders yourself you need to follow these steps.
After installing the required packages you can run the experiments by first pretraining the autoencoders by calling:

> python pretrain_autoencoders.py --json experiment_setups/pretraining_{data}.json

where ```{data}``` should be replaced with ```uci```, ```synthetic``` or ```images``` depending on which data sets you want to train on. This will automatically pretrain ten autoencoders on the different data sets and will save them in ```results/{data set name}/pretrained_aes```.

After the pretraining finished (or you used our pretrained models by downloading them from zenodo and moved them to the results directory), you can perform DECCS in the next step.

## Run DECCS

To run DECCS for the different data sets you can use the predefined experiment settings in the experiment_setups directory, e.g. to run DECCS on the image data sets you need to run:

> python run_deccs.py --json "experiment_setups/deccs_images.json"

This will save the results under results/{data set name}/{deccs-version}. In the json files in the experiment_setup directory the setting with the reconstruction loss is set to the default. If you want to run the experiment without the reconstruction loss, then you need to set ```use_rec:false``` in the corresponding json file.

# Citation (will be replaced with ICDM citation)
```
@article{miklautz2022deep,
  title={Deep Clustering With Consensus Representations},
  author={Miklautz, Lukas and 
          Teuffenbach, Martin and 
          Weber, Pascal and 
          Perjuci, Rona and 
          Durani, Walid and 
          B{\"o}hm, Christian and
          Plant, Claudia},
  journal={arXiv preprint arXiv:2210.07063},
  year={2022}
}
```


