# CompeteSMoE - Effective Training of Sparse Mixture of Experts via Competition
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper [CompeteSMoE - Effective Training of Sparse Mixture of Experts via Competition]()</br>
Our implementation is based on the [Sandwich Transformer](https://github.com/ofirpress/sandwich_transformer). More training scripts and datasets are coming soon. 

## Prerequisites
- [FastMoE](https://github.com/laekov/fastmoe): A fast MoE impl for PyTorch

## Running Experiments in the Paper

#### Pre-training
- Download the enwik8 dataset from [here](https://drive.google.com/drive/folders/1IFwCSf9JSeyviDGw5tyHmrdxyUiqsjFt?usp=drive_link), then put it into the directory `datasets/pretraining/`</br>
```bash
datasets/
└── pretraining
    └── enwik8
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

- Select the Transformer architecture, its scale, and the type of SMoE layer. We support:

|   | SMoE | SMoE-Dropout | XMoE | StableMoE |
|---|---|---|---|---|
| Transformer (S/M/L) | <input type="checkbox" disabled checked /> |  |  |  |
| Brainformer (S/M/L) |  |  |  |  |

- Run the corresponding script. For example, run the below command to pre-train the Transformer at a small scale with SMoE layers. </br>
`bash scripts/pretraining/enwik8/transformers/smoe-s.sh`

- The checkpoint will be saved at `checkpoints/enwik8/transformers-s` during training. 

## Citation