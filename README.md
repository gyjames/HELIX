# HELIX

**HELIX** is a deep learning model that integrates pre-mRNA sequence and RNA-binding protein (RBP) expression profiles to predict tissue- and condition-specific splicing patterns and transcript isoform usage.

## Installation

- HELIX is a deep learning model constructed based on PyTorch 2.1.0 and Python 3.11.5.

- We recommend you install dependencies using conda:
```
conda create -n helix python=3.11.5 pytorch pandas numpy pyfaidx
```
- If you are running HELIX on GPU, **CUDA** is required.

```
- It takes 2-10 seconds to finish prediction for ~100 transcripts (depends on the GPU).
- It takes 7-10 days to train the whole model on one GTX 3080.

## Usage

**Input format** Generate input files upon transcript annotations / gene expression matrix provided with script/preprocessing.py
- In the provided gene matrix, each column represents a sample and each row represents a gene. See the format in /demo.
- The preprocessing step will generate two input txt files for splice site model and transcript model, respectively, as well as a normalized rbp expression in .pickle and .tsv.
- Information of transcript unit for subsequent isoform usage prediction is in *tss_group.tsv*. 
- For customized splice site prediction (not derived from gtf annotation), see the input file format in /demo.

Splice site input file format:

| Identifier | Chromosome | Strand | Gene | Splice site type | Location | Sample | Label1 | Label2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1	| chr1	| +	| ENSG00000237491.10_10	| a	| 728262	| Adipose_Subcutaneous	| 0	| 0 |

- Identifier must be unique
- Gene, Splice site type, Label1 and Label2 are not necessary provided
- Sample must keep same with rbp expression file




```
python script/preprocessing.py -g annotation.gtf -o out_dir -r gene_tpm.mtx
```

**Step 2.** Run HELIX.py to simutaneously predict splicing strength and isoform usage for annotated transcript isoforms.

- Pretrained model weights have been uploaded in the script/model directory
- The input files (two txt and rbp expression) are generated through the preprocessing step.

```
python HELIX.py -ds demo/splice_site_input.txt -dt demo/transcript_input.txt -rbp demo/rbp.pickle -o outputdir -d 'cuda:0' -g reference.fa
```

- If only splicing strength is needed, use the parameter *--ssonly*

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
