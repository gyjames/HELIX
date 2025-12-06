# HELIX

**HELIX** is a deep learning model that integrates pre-mRNA sequence and RNA-binding protein (RBP) expression profiles to predict tissue- and condition-specific splicing patterns and transcript isoform usage.

## Features

- HELIX give probabilities splice sites being acceptor or donor, splicing strength, splicing regulatory level and probability of being upregulated/downregulated under given RBP expression condition
- HELIX give isoform usage within predefined transcription units (transcripts using proximal transcription start sites)

## Installation

- HELIX is a deep learning model constructed based on PyTorch 2.1.0 and Python 3.11.5.

- We recommend you install dependencies using conda:
```
conda create -n helix python=3.11.5 pytorch pandas numpy pyfaidx
```
- If you are running HELIX on GPU, **CUDA** is required.

- It takes 2-10 seconds to finish prediction for ~100 transcripts (depends on the GPU).
- It takes 7-10 days to train the whole model on one GTX 3080.

## Input format

### RBP expression

- RBP expression file is a gene expression matrix in TPM. Each row represents a gene, each column represents a sample. Genes are required to use Ensembl gene identifiers (ENSG format).

### Splice sites list

- Splice site model input. Format is shown below:

| Identifier | Chromosome | Strand | Gene | Splice site type | Location | Sample | Label1 | Label2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1	| chr1	| +	| ENSG00000237491.10_10	| a	| 728262	| Adipose_Subcutaneous	| 0	| 0 |

- Each line represents a splicing event to predict

### Transcript model input

- Each line represents a transcript to predict

## Quick start


- Predict splicing strength and isoform usage

```
python HELIX.py -ds demo/splice_site_input.txt -dt demo/transcript_input.txt -rbp demo/rbp.pickle -o outputdir -d 'cuda:0' -g reference.fa
```

- Predict splicing strength only

```
python HELIX.py -ds demo/splice_site_input.txt -rbp demo/rbp.pickle -o outputdir -d 'cuda:0' -g reference.fa --ssonly
```

- When predicting splice with single cell RNA-seq data (10X), use the parameter *--sc*

```
python HELIX.py -ds demo/splice_site_input.txt -dt demo/transcript_input.txt -rbp demo/rbp.pickle -o outputdir -d 'cuda:0' -g reference.fa --sc
```

See full options below:
```
options:
  -h, --help            show this help message and exit
  -ds SSINPUT, --ssinput SSINPUT
                        Input for splice site model.
  -dt TXINPUT, --txinput TXINPUT
                        Input for transcript model.
  -rbp RBPINPUT, --rbpinput RBPINPUT
                        Normalized RBP path.
  -o OUT, --out OUT     Output directory.
  -g GENOME, --genome GENOME
                        Reference genome path.
  -d DEVICE, --device DEVICE
                        Device (CPU or GPU index)
  -c CORE, --core CORE  Number of CPU core. Default 1.
  -bs BATCHSIZES, --batchsizes BATCHSIZES
                        Batch size for splice site model prediction. Default 64.
  -bt BATCHSIZET, --batchsizet BATCHSIZET
                        Batch size for transcript model prediction. Default 32.
  --ssonly              Only predict splicing strength.
  --sc                  Predict splicing strength and isoform usage with RBP expression derived from 10X data.

```

## Output

- Output of splice site model has 11 columns: splice site index, splice site type(derived from gtf annotation), probability of being acceptor, probability of being donor, acceptor splicing strength (baseline), donor splicing strength (baseline), acceptor splicing regulatory level, donor splicing regulatory level, probability of no regulation, probability of upregulation, probability of downregulation
- Output of transcript model has 2 columns: transcript index, isoform usage

## Changelog

[v1.1] 2025-11-11
- Fix: fix some bugs.
