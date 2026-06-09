# UMAP-Based Representation Learning and Clustering for Exoplanet Population Structure Discovery

### Investigating Nonlinear Manifold Structure in High-Dimensional Exoplanet Data

---

## Scientific Motivation

The discovery of thousands of exoplanets has transformed planetary science into a data-rich field. As observational surveys continue to expand, understanding how planetary systems are organized within high-dimensional parameter space has become an increasingly important challenge.

Traditional approaches often rely on predefined classifications or linear statistical techniques, which may fail to capture complex nonlinear relationships among planetary and stellar properties.

This project investigates whether exoplanet populations exhibit an underlying nonlinear manifold structure that can be recovered through unsupervised representation learning.

Rather than focusing only on prediction, the goal is scientific discovery: uncovering latent geometric organization within exoplanet datasets and evaluating whether this organization corresponds to physically meaningful planetary regimes.

---

## Research Question

> Do exoplanet populations exhibit stable nonlinear manifold structure that can be recovered through unsupervised representation learning methods beyond traditional linear dimensionality reduction?

---

## Main Scientific Finding

The results suggest that exoplanets are not uniformly distributed throughout parameter space.

Across multiple experiments and embedding methods, UMAP consistently reveals stable low-dimensional structure that is significantly less apparent under linear projection methods such as PCA.

The findings provide evidence that exoplanet populations occupy partially separable nonlinear regions shaped by relationships among planetary and stellar properties.

Furthermore, incorporating domain-informed constraints improves astrophysical interpretability, highlighting the potential of physics-aware machine learning for scientific discovery.

---

## Key Contributions

### Representation Learning Framework

* Principal Component Analysis (PCA) baseline
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Uniform Manifold Approximation and Projection (UMAP)
* Physics-informed UMAP-style audit extension

### Scientific Evaluation Framework

* Geometric clustering assessment
* Embedding robustness analysis
* Multi-seed reproducibility testing
* Physics-consistency evaluation

### Scientific Insights

* Discovery of latent exoplanet population structure
* Comparison of linear and nonlinear representations
* Analysis of geometry-versus-physics trade-offs
* Reproducible scientific machine-learning workflow

---

## Methodology

```text
Exoplanet Catalog
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
Representation Learning
(PCA / t-SNE / UMAP)
        │
        ▼
Latent Embedding Space
        │
        ▼
Clustering Analysis
(K-Means)
        │
        ▼
Robustness Evaluation
        │
        ▼
Scientific Interpretation
```

---

## Dataset Features

The framework operates on astrophysical variables including:

* Planet mass
* Planet radius
* Orbital period
* Semi-major axis
* Equilibrium temperature
* Stellar mass
* Stellar radius
* Stellar effective temperature

Additional planetary and stellar parameters may be incorporated depending on catalogue availability.

---

## Evaluation Metrics

To assess both geometric quality and scientific relevance, multiple evaluation criteria are employed.

| Metric | Purpose |
| --- | --- |
| Silhouette Score | Measures cluster separability |
| Davies--Bouldin Index | Measures cluster compactness |
| Physics Consistency Score (PCS) | Measures astrophysical interpretability |

---

## Physics Consistency Score (PCS)

Traditional clustering metrics evaluate geometric structure but do not assess scientific meaning.

To address this limitation, a Physics Consistency Score (PCS) is introduced.

PCS evaluates whether discovered clusters correspond to distinct physical regimes by measuring cluster-level variation across key astrophysical parameters.

### Interpretation

* PCS ≈ 0 → little physical distinction between clusters
* Higher PCS → stronger alignment with astrophysical structure
* High PCS clusters are more likely to represent meaningful planetary populations

This metric complements traditional clustering measures by emphasizing scientific relevance rather than purely geometric separation.

---

## Results

### Embedding Performance

| Method | Silhouette ↑ | Davies--Bouldin ↓ | Physics Consistency ↑ |
| --- | --- | --- | --- |
| PCA | 0.33 | 1.00 | 0.10 |
| t-SNE | 0.40 | 0.79 | 0.15 |
| UMAP | **0.48** | **0.69** | 0.28 |
| Physics-UMAP | 0.46 | 0.84 | **0.35** |

---

## Key Findings

### Nonlinear Methods Outperform Linear Projections

PCA struggles to separate exoplanet populations due to its linear assumptions.

Both UMAP and t-SNE reveal structure that is largely hidden in linear embeddings.

### UMAP Produces the Most Balanced Representation

UMAP achieves the strongest overall performance by simultaneously preserving local relationships and maintaining meaningful global organization.

### Stable Across Random Initializations

Repeated experiments demonstrate low variance across embedding runs, indicating that the recovered structure is not simply an artifact of stochastic optimization.

### Geometry and Physics Are Not Identical Objectives

Physics-informed embeddings improve astrophysical interpretability while slightly reducing geometric compactness, revealing an important trade-off between mathematical clustering quality and scientific meaning.

---

## UMAP Stability Analysis

| Metric | Average |
| --- | --- |
| Silhouette Score | ~0.495 |
| Davies--Bouldin Index | ~0.67 |
| Variability Across Seeds | Low |

The results indicate strong reproducibility and stability of the learned manifold structure.

---

## Visual Comparison

### PCA Representation

![PCA embedding](figures/readme/pca_embedding.png)

### UMAP Representation

![UMAP embedding](figures/readme/umap_embedding.png)

### t-SNE Representation

![t-SNE embedding](figures/readme/tsne_embedding.png)

### PCA, UMAP, and t-SNE Representation Comparison

This repository includes a combined visual comparison of PCA, UMAP, and t-SNE embeddings.

![PCA, UMAP, and t-SNE comparison](figures/readme/pca_umap_tsne_comparison.png)

### UMAP Representation with KNN Decision Geometry

![UMAP representation with KNN decision geometry](figures/readme/umap_knn_geometry.png)

### Habitable-Subset UMAP Representation

![UMAP representation of conservative and optimistic habitable candidates](figures/readme/umap_habitable_subset.png)

### Physics-Audit Distribution

![Physics-informed habitability distribution](figures/readme/physics_habitability_distribution.png)

---

## Scientific Significance

This work contributes to the growing field of Scientific Machine Learning (SciML), where machine learning is used not only for prediction but also for understanding scientific systems.

The project demonstrates how manifold learning can be used to:

* Explore exoplanet population structure
* Discover latent organization in astrophysical datasets
* Generate interpretable scientific hypotheses
* Investigate nonlinear physical relationships
* Support data-driven scientific discovery

---

## Potential Research Impact

The proposed framework may contribute to future research in:

* Exoplanet population studies
* Planet formation regime identification
* Habitability space exploration
* Rare planetary system discovery
* Physics-informed representation learning
* Machine learning for scientific discovery

Because the methodology is domain-agnostic, it can be extended to other scientific fields involving high-dimensional physical systems.

---

## Limitations

* No ground-truth astrophysical population labels
* Dependence on embedding hyperparameters
* Limited astrophysical validation
* Exploratory interpretation of discovered clusters

---

## Future Work

### Machine Learning Extensions

* HDBSCAN clustering
* Contrastive representation learning
* Variational autoencoders
* Self-supervised learning
* Graph neural networks

### Scientific Extensions

* Planet formation regime analysis
* Habitability manifold exploration
* Uncertainty-aware embeddings
* Physics-constrained representation learning
* Integration with theoretical planetary evolution models

---

## Reproducibility

```bash
git clone https://github.com/yourusername/exoplanet-umap-analysis.git
cd exoplanet-umap-analysis
pip install -r requirements.txt
python main.py
```

Core reusable modules include:

* `preprocess.py` — catalogue loading, feature mapping, scaling inputs
* `embedding.py` — PCA and repeated nearest-neighbour representation evidence
* `clustering.py` — lightweight clustering diagnostics
* `evaluation.py` — physics consistency, PAGER scoring, and ranking metrics

---

## Research Areas

* Scientific Machine Learning
* Unsupervised Learning
* Representation Learning
* Manifold Learning
* Computational Astrophysics
* Exoplanet Science
* Explainable AI
* Physics-Informed Machine Learning

---

## Conclusion

This study provides evidence that exoplanet populations exhibit latent nonlinear structure that is difficult to observe using traditional linear methods.

Among the investigated approaches, UMAP offers the strongest balance between clustering quality, stability, and physical interpretability. The results suggest that manifold learning provides a powerful framework for uncovering hidden organization within exoplanet parameter space and may serve as a useful tool for future astrophysical discovery.

More broadly, the project demonstrates how Scientific Machine Learning can be used not only to predict outcomes but also to reveal previously unobserved structure in complex physical systems.

---

## Citation

```bibtex
@software{worku2026umap_exoplanets,
  author = {Samuel Worku},
  title = {PAGER: A Physics-Audited Exoplanet Follow-up Prioritisation Framework},
  year = {2026},
  url = {https://github.com/yourusername/exoplanet-umap-analysis}
}
```
