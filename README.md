# HELIX

**HELIX** is a deep learning model that integrates pre-mRNA sequence and RNA-binding protein (RBP) expression profiles to predict tissue- and condition-specific splicing patterns and transcript isoform usage.

## Installation

- HELIX is a deep learning model constructed based on PyTorch 2.1.0 and Python 3.11.5.

- Pytorch installation: https://pytorch.org/get-started/locally/

- Other dependencies can be installed via conda and pip:

```
conda create -n helix python=3.11.5
pip install pandas pickle numpy pyfaidx
```

## Usage

1. Generate input files upon transcript annotations / gene expression matrix provided.

```
python scripts/read_annotation.py annotation.gtf
```

2. Run HELIX.py to simutaneously predict splicing strength and isoform usage for annotated transcript isoforms. 

```
python helix.py RBP.tsv annotation.db
```
