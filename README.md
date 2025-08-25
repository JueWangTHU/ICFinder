# ICFinder: Ion channel identification and ion permeation residue prediction

This repo holds the code of BLAPE (Bi-LSTM and Pre-trained Encoder) for ion channel identification. 

And the code of CLAPE (Contrastive Learning And Pre-trained Encoder) for ion permeation residue prediction. 

The CLAPE framework has been applied to [DNA-binding](https://github.com/YAndrewL/clape) and [small molecule-binding](https://github.com/JueWangTHU/CLAPE-SMB) prediction.

The protein language model used in this study is [ESM-2](https://github.com/facebookresearch/esm).

[ICFinder webserver:]()

## Environment
```
conda env create -f environment.yml
```
## Files and folders description
### 1. Sequence
This folder contains protein sequences for training and inference input. 

We use 3 lines to describe each sequence.
```
>ID
Seq
Label (if needed)
```
Labels are required for training the model but not for inference.

#### ICIdentification
Training, validation, and testing sets for ion channel identification.

#### CaICIdentification
Training, validation, and testing sets specifically for calcium channel identification.

#### ICPermeation
Training, validation, and testing sets for ion permeation residue prediction.

#### CaICPermeation
Training, validation, and testing sets specifically for calcium ion permeation residue prediction.

#### UniRef50
The dataset can be downloaded using the following command:
```
wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip uniref50.fasta.gz
```

### 2. Embedding
This folder stores embeddings generated from sequences. 

### 3. Model
This folder contains the models trained in this study.

### 4. inference.py
Directly use our models.
#### Usage:
```
python inference.py <input_file> <output_file>
```
Ion channel identification and ion permeation residue prediction from protein sequences.

Here we provide an example, please use following commands:

```
python inference.py Sequence/example.txt result_example.txt
```

### Train your own models. 
### 5. pre.py
This python file generates protein sequence embeddings. 

### 6. BLAPE
```
cd BLAPE/
```
### 6.1 train.py
Train your own BLAPE models with new datasets and hyperparameters. Log files are saved in the `train_log/` folder.

### 6.2 IG.py
Interpretability analysis based on integrated gradients (IG) was performed to identify which residues the model focuses on when predicting a sequence as an ion channel. Log files are saved in the `IG_log/` folder.

### 7. CLAPE
```
cd CLAPE/
```
### 7.1 triplet.py
Train your own CLAPE models with new datasets and hyperparameters. Log files are saved in the `triplet_classification/` folder.

The modules `data.py`, `losses.py`, and `model.py` are not executed independently, but are utilized by `triplet.py`.

Please contact wangjue21@mails.tsinghua.edu.cn for questions. 
