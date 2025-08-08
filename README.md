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
- In the provided gene matrix, each column represents a sample and each row represents a gene. See the format in /demo.
- The preprocessing step will generate two input txt files for splice site model and transcript model, respectively, as well as a normalized rbp expression .pickle file. 

```
python scripts/read_annotation.py -g annotation.gtf -o out_dir -r gene_tpm.mtx
```

2. Run HELIX.py to simutaneously predict splicing strength and isoform usage for annotated transcript isoforms.

- Pretrained model weights have been uploaded in the model/ directory
- The input files (two txt and rbp expression) are generated through the preprocessing step.  

```
python HELIX.py -b models/baseline.pth -r models/regulatory.pth -t models/tx.pth -ds demo/splice_site_input.txt -dt demo/tx_input.txt -rbp demo/rbp.pickle -o outputdir -c 'cuda:0'
```

See full options below:
```
options:
  -h, --help            show this help message and exit
  -b BASELINE, --baseline BASELINE
                        Baseline module path.
  -r REGULATORY, --regulatory REGULATORY
                        Regulatory module path.
  -t TRANSCRIPT, --transcript TRANSCRIPT
                        Transcript model path.
  -ds SSINPUT, --ssinput SSINPUT
                        Input for splice site model.
  -dt TXINPUT, --txinput TXINPUT
                        Input for transcript model.
  -rbp RBPINPUT, --rbpinput RBPINPUT
                        Normalized RBP path.
  -o OUT, --out OUT     Output directory.
  -c DEVICE, --device DEVICE
                        Device
```

## Output

- Output of splice site model has 11 columns: splice site index, splice site type(derived from gtf annotation), probability of being acceptor, probability of being donor, acceptor splicing strength (baseline), donor splicing strength (baseline), acceptor splicing regulatory level, donor splicing regulatory level, probability of no regulation, probability of upregulation, probability of downregulation
- Output of transcript model has 2 columns: transcript index, isoform usage
