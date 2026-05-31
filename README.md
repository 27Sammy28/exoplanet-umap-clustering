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

This indicates that geometric cluster quality and physical interpretability are not always aligned, highlighting the importance of multi-objective evaluation in scientific machine learning.
---

## Scientific Significance

This work contributes to scientific machine learning, where the goal is not only prediction but also understanding structure in complex datasets.

In the context of exoplanet science, this approach provides a complementary perspective to traditional classification by focusing on latent geometric structure in parameter space.

## Scientific Insight

Beyond evaluating clustering performance, this work provides insight into the underlying structure of exoplanet parameter space.

The results suggest that exoplanet populations are not uniformly distributed in feature space but instead form partially separable regions that are more clearly revealed through nonlinear embeddings. In particular, UMAP consistently uncovers structured groupings that are not as apparent under linear projection (PCA), indicating the presence of nonlinear relationships between physical planetary properties.

A key observation is that clustering stability remains relatively consistent across multiple embedding runs, suggesting that the discovered structure is not purely an artifact of initialization but reflects persistent geometric relationships in the data.

However, the comparison between geometric clustering quality and physics-informed consistency shows an important trade-off: improving alignment with physical properties can slightly reduce cluster compactness. This indicates that purely geometry-driven optimization may not fully align with physically meaningful structure, motivating the need for hybrid approaches that incorporate domain constraints.

Overall, the findings support the hypothesis that exoplanet populations exhibit latent manifold structure that can be partially recovered using nonlinear representation learning methods.

## Physics Consistency Score (Definition)

To evaluate whether clusters correspond to physically meaningful groupings, a Physics Consistency Score (PCS) is computed.

This metric measures how strongly clustered groups align with key physical attributes of exoplanets, such as radius, mass, orbital period, and equilibrium temperature.

### Definition

For each cluster:

1. Compute the mean of each physical feature within the cluster
2. Compare cluster-level feature variance against global dataset variance
3. Measure how much each cluster deviates from the overall population structure

The final score is defined as:

- Higher values indicate that clusters correspond to more distinct physical regimes
- Lower values indicate clusters that are not meaningfully different in physical space

### Interpretation

- PCS ≈ 0 → clusters do not reflect physical structure  
- PCS → higher values indicate stronger alignment with astrophysical properties  
- PCS is used as a complementary metric to geometric clustering scores (Silhouette, Davies-Bouldin)

### Purpose

Unlike standard clustering metrics, Physics Consistency is designed to evaluate whether learned representations preserve scientifically meaningful structure rather than purely geometric separability.

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

## Conclusion

This work demonstrates that nonlinear representation learning methods, particularly UMAP, can uncover stable and structured organization in high-dimensional exoplanet parameter spaces. Across multiple evaluation criteria—including clustering quality, stability across runs, and physics-consistency alignment—UMAP consistently outperforms linear baselines such as PCA and provides more globally coherent structure than t-SNE.

A key finding is that the learned structure is not only geometrically stable but also partially aligned with underlying physical properties of exoplanets, suggesting that the dataset contains intrinsic nonlinear relationships that are better captured through manifold learning techniques.

However, the results also highlight an important trade-off between geometric clustering quality and physical interpretability. While physics-informed embeddings improve alignment with domain-relevant features, they may slightly reduce compactness in latent space. This suggests that future approaches should consider hybrid objectives that jointly optimize geometric structure and physical constraints.

Overall, the results support the hypothesis that exoplanet populations exhibit latent nonlinear manifold structure that can be partially recovered through unsupervised representation learning, making this a promising direction for scientific machine learning in astrophysical discovery tasks.

