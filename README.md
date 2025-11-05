# HMCC  

## Overview  

**HMCC** is a neural network framework designed for multi-label prediction on microbial protein sequences. It integrates **ProtBert embeddings**, **Focal Loss with re-weighting**, and **frequency-aware sampling strategies** to address challenges posed by long-tail label distributions.  
For **Gene Ontology (GO) embedding**, refer to: [OWL2Vec-Star](https://github.com/KRR-Oxford/OWL2Vec-Star.git)  or you can refer to cluster_code.

![workflow](./image/workflow.png)

---

## Data
- The dataset used in this project is hosted on Google Drive and can be accessed via the following link: https://drive.google.com/file/d/1Nk08PhCMnCZ3IAkU9eyNKKPV8KjZwLp6/view?usp=sharing

Tested environment:

- Python 3.10

- PyTorch 2.3

- CUDA 12.1

- Transformers ≥ 4.40

- Accelerate ≥ 0.30


## Installation  

- Install dependencies using:  

```bash  
pip install -r requirements.txt  
```  

- If you encounter version issues, install the tested versions:

```bash 
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
```

---

## Pretrained Models and Data

1. Embedding Models

- [protbert](https://hf-mirror.com/Rostlab/prot_bert)

- [esm2_650M](https://hf-mirror.com/facebook/esm2_t33_650M_UR50D)

- [sapbert](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)


2. HMCC Pretrained Weights (optional)

- We provide pretrained weights for direct evaluation: hmcc_pretrained.pth (available in https://drive.google.com/file/d/1tSbGHRavOrw51lg__Spx7u_R9wPnqcJJ/view?usp=drive_link).

3. Datasets

- Protein sequences and GO annotations are based on UniProt (date: 2024-03 release).

- GO ontology file: go.obo (download from [Gene Ontology](https://geneontology.org/docs/download-ontology/)
).

HMCC/
 ├── data/
 │   ├── sequences.fasta
 │   ├── go.obo
 │   └── annotations.tsv
 ├── checkpoints/
 ├── cluster_code/
 ├── freq/
 ├── image/
 └── train.py

# Reproducibility

(1) GO Embedding and Clustering

```
# Generate SapBERT-based embeddings
python cluster_code/bert_embedding.py

# Cluster the GO vectors
python cluster_code/GAN_code.py

# (Optional) Visualize with t-SNE
python cluster_code/visual_namespace.py
```

(2) Protein Sequence Embedding

```
python data2milvus.py
```

- Edit the model path in data2milvus.py to choose between ProtBert or ESM2.

(3) Frequency Re-weighting

```
python freq/freqs.py
python freq/get_high_low_indices.py
```

(4) Train HMCC

```
accelerate launch --main_process_port 29501 \
                  --config_file accelerate_config.yaml \
                  --mixed_precision bf16 \
                  train.py >> Prot_milvus.log
```

or Train DeepGOPlus

```
python deepgoplus_train.py
```

(5) Evaluation and Expected Results

```
python eval.py
```

## Citation

If you use HMCC in your work, please cite:

```
@article{chen2025hmcc,
  title={Hierarchical Cascaded Context Network for Microbial Protein Function Prediction},
  author={Chen, Shengyang and Yang, Yuqing and et al.},
}
```
