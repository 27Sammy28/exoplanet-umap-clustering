# UMAP-Based Representation Learning and Clustering for Exoplanet Population Structure Analysis

---

## Abstract

This project investigates whether nonlinear manifold learning can uncover stable and physically meaningful structure in high-dimensional exoplanet datasets. I apply Uniform Manifold Approximation and Projection (UMAP) to construct low-dimensional representations of astrophysical features, followed by unsupervised clustering to analyze latent population structure.

I evaluate whether these representations preserve meaningful separability of exoplanet groups and whether clustering remains stable under variations in embedding parameters and initialization. I also compare nonlinear methods (UMAP and t-SNE) against linear baselines (PCA) to assess their effectiveness in capturing structure in scientific datasets.

---

## Research Objective

Exoplanet datasets contain complex nonlinear relationships between physical variables such as mass, radius, orbital period, and temperature. Traditional linear projection methods often fail to capture these relationships.

This work investigates the following research question:

> Do exoplanet populations exhibit a stable nonlinear manifold structure that can be discovered through unsupervised representation learning methods beyond linear dimensionality reduction?

---

## Key Contributions

This project provides:

- A complete UMAP-based representation learning pipeline for exoplanet data  
- A clustering framework for latent structure discovery  
- A robustness analysis across multiple embedding runs  
- A quantitative comparison of PCA, t-SNE, and UMAP  
- A physics-consistency evaluation of learned representations  
- A reproducible workflow for scientific machine learning experiments  

---

## Methodology

### Data Processing
- Feature selection from exoplanet catalog attributes  
- Handling missing values  
- Normalization and scaling of physical features  

### Representation Learning
- PCA (linear baseline)  
- t-SNE (local structure baseline)  
- UMAP (nonlinear manifold learning)  
- Physics-informed UMAP variant  

### Clustering
- K-Means clustering in embedded space  
- Cluster assignment across different methods  

### Robustness Analysis
- Multi-seed stability evaluation  
- Sensitivity analysis of UMAP parameters  
- Cluster consistency across runs  

---

## Evaluation Metrics

The performance of each method is evaluated using:

- **Silhouette Score** (higher is better)  
- **Davies–Bouldin Index** (lower is better)  
- **Physics Consistency Score** (higher is better, measures alignment with physical structure)  

---

## Results

### Main Comparison

| Method | Silhouette ↑ | Davies-Bouldin ↓ | Physics Consistency ↑ |
|--------|--------------|------------------|------------------------|
| PCA | 0.33 | 1.00 | 0.10 |
| t-SNE | 0.40 | 0.79 | 0.15 |
| UMAP | 0.48 | 0.69 | 0.28 |
| Physics-UMAP | 0.46 | 0.84 | 0.35 |

---

### Interpretation

- PCA fails to capture nonlinear structure due to its linear nature  
- t-SNE improves local clustering but lacks global consistency  
- UMAP achieves the best balance between separability and structure preservation  
- Physics-UMAP improves interpretability but slightly reduces clustering compactness  

---

### UMAP Stability Analysis

Across multiple runs, UMAP shows consistent clustering behavior:

- Mean Silhouette Score ≈ **0.495**  
- Stable Davies–Bouldin Index (~0.67–0.68 range)  
- Low variance across different random seeds  

This indicates strong embedding stability and reproducibility.

---

### Representation Learning Comparison

| PCA | UMAP | t-SNE |
|-----|------|-------|
| (see figure) | (see figure) | (see figure) |

**Observations:**

- PCA does not separate structure effectively  
- UMAP reveals clearer and more coherent latent grouping  
- t-SNE emphasizes local structure but distorts global relationships  

---

## Scientific Significance

This work contributes to scientific machine learning, where the goal is not only prediction but also understanding structure in complex datasets.

In the context of exoplanet science, this approach provides a complementary perspective to traditional classification by focusing on latent geometric structure in parameter space.

---

## Limitations

- Clustering is unsupervised and not validated against ground-truth astrophysical labels  
- Results depend on UMAP hyperparameters  
- Further astrophysical validation is required for scientific interpretation  

---

## Future Work

- Incorporate HDBSCAN for density-based clustering  
- Add physics-informed feature engineering  
- Improve uncertainty estimation in embeddings  
- Extend to graph-based representation learning  
- Perform deeper astrophysical validation of clusters  

---

## Reproducibility

This project is designed for reproducible scientific experimentation.

Future improvements will include:

- modular `src/` structure  
- fixed random seeds  
- configuration-driven experiments  
- standardized evaluation pipeline  
- full environment specification (`requirements.txt`)  

---

## Research Direction

This work aligns with:

- Scientific machine learning  
- Unsupervised representation learning  
- Manifold learning in physical systems  
- High-dimensional astrophysical data analysis  
- Exoplanet population structure discovery  
