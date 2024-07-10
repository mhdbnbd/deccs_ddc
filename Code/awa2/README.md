# AwA2 Dataset Processing with Autoencoder and KMeans Clustering

## Overview

This project processes the Animals with Attributes 2 (AwA2) dataset by creating a sample dataset, training an autoencoder model, extracting embeddings, and performing KMeans clustering on the embeddings. The script includes functionality to run with or without GPU support.

## Directory Structure

awa2/
│
├── data/
│   └── AwA2-data/
│       └── Animals_with_Attributes2/
│           ├── JPEGImages/
│           └── AwA2-labels.txt
├── awa2/
│   ├── main.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── README.md



## Installation

1. Clone the repository:
   ```bash
   git clone 
   cd awa2  
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt