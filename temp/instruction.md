# CLUSTERING INSTRUCTION

## Obtaining OBO Files

To obtain the latest OBO file, you can download it from the [Gene Ontology website](http://geneontology.org/).

## Clustering Methodology

In our project, we employ **Gene Ontology (GO) embedding** techniques. For implementation details, please refer to:  
[OWL2Vec-Star Repository](https://github.com/KRR-Oxford/OWL2Vec-Star.git)  
We also provide simple implementations of Graph Neural Networks (GNNs) and Generative Adversarial Networks (GANs) for generating ontology embeddings.

## Code File Descriptions

### Scripts:
- **`bert_embedding.py`**  
  Generates:  
  - `go_sapbert_cls_embeddings.npy` (embedding vectors)  
  - `go_sapbert_ids.csv` (corresponding GO term IDs)

- **`GNN_code.py`**  
  Training code for graph neural networks.

- **`visual_*.py`**  
  Visualization utilities (multiple variants available).
