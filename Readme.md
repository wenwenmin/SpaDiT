# SpaDiT: Diffusion Transformer for Spatial Gene Expression Prediction using scRNA-seq

A novel deep learning method that uses a diffusion generative model to integrate scRNA-seq data and
ST data for the prediction of undetected genes

## Overview of SpaDiT

![](model/model.png)

## Setup

```
pip install -r requirement.txt
```

## Preprocess data

To preprocess the data, run the `data_preprocess.py` script located in the `preprocess` directory. Use the following command:

```
python preprocess/data_preprocess.py --input data/raw_data.csv --output data/processed_data.csv
```

## Running Experiments

To train the neural network, use the following command:

```
python main.py --config config/train_config.json
```

To evaluate the model, run:

```
python evaluate.py --model_path models/best_model.pth --data_path data/test_data

```

