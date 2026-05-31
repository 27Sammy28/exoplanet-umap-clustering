UMAP-Based Representation Learning and Clustering for Exoplanet Population Structure Analysis

# Abstract

This project investigates whether nonlinear manifold learning can uncover stable and physically meaningful structure in high-dimensional exoplanet parameter spaces. We apply Uniform Manifold Approximation and Projection (UMAP) to construct low-dimensional embeddings of astrophysical features, followed by unsupervised clustering to analyze latent population structure.

We evaluate whether learned representations preserve meaningful separability of exoplanet populations and whether clustering structure remains robust under variations in embedding configuration and clustering initialization. The study further compares nonlinear embeddings against linear baselines such as PCA to assess the advantage of manifold-based representations in astrophysical data analysis.


# Research Objective

Exoplanet catalogs contain high-dimensional, noisy, and partially correlated physical parameters (e.g., mass, radius, orbital period, equilibrium temperature). Traditional linear methods often fail to capture nonlinear dependencies in this space.

This work addresses the following research question:

Do exoplanet populations exhibit stable nonlinear manifold structure that enables unsupervised discovery of physically meaningful groupings beyond linear projection methods?

# Key Contributions

This project contributes:

A UMAP-based representation learning pipeline for exoplanet feature spaces
A clustering framework for latent space structure discovery
A robustness analysis of embedding stability under parameter variation
A comparative evaluation against linear dimensionality reduction (PCA)
A reproducible workflow for scientific machine learning on astrophysical datasets



#  Methodology

## Data Processing
Feature selection from exoplanet catalog attributes
Handling of missing values and inconsistent measurements
Normalization and scaling of physical parameters

## Manifold Learning
Uniform Manifold Approximation and Projection (UMAP)
Exploration of neighborhood size and embedding dimension sensitivity

## Unsupervised Structure Discovery
K-Means clustering in latent space
Cluster assignment analysis across embedding configurations

## Robustness and Validation
Stability analysis across multiple random seeds
Sensitivity testing with UMAP hyperparameters
Qualitative cluster consistency evaluation

##  Baseline Comparison
Principal Component Analysis (PCA)
Qualitative comparison of separability and structure preservation


# Technologies
Python
NumPy
Pandas
Scikit-learn
UMAP-learn
Matplotlib
Seaborn


#  Experimental Design

To evaluate the reliability of learned structure, the following analyses are performed:

Embedding stability under random initialization
Cluster consistency across different UMAP configurations
Comparative structure analysis between PCA and UMAP embeddings
Visual separability assessment in latent space


# Results (Ongoing / Preliminary)

Preliminary results suggest that nonlinear embeddings produced by UMAP reveal more structured and separable groupings in exoplanet parameter space compared to linear projection methods. Cluster formation shows partial stability under parameter variation, indicating the presence of underlying manifold structure in the dataset.

Further quantitative validation is ongoing.

#  Scientific Significance

This work contributes to the growing field of scientific machine learning, where representation learning is used to uncover structure in observational scientific datasets. In the context of exoplanet science, this approach provides a complementary perspective to traditional classification methods by focusing on latent geometric structure rather than predefined labels.


# Limitations
Clustering results are unsupervised and not directly tied to astrophysical ground truth classes
Sensitivity to UMAP hyperparameters remains a challenge
Further domain-specific validation with astrophysical constraints is required
9. Future Work
Integration of HDBSCAN for density-based clustering
Physics-informed feature engineering
Temporal analysis of exoplanet discovery datasets
Incorporation of uncertainty-aware embedding methods
Extension to graph-based representation learning


# Reproducibility

This repository is designed for reproducible scientific experimentation.

Future updates will include:

modular Python source code (src/)
configuration-driven experiments
fixed random seed pipelines
standardized evaluation scripts
full environment specification (requirements.txt)


# Research Direction

This project aligns with research in:

Scientific machine learning
Unsupervised representation learning
High-dimensional data analysis
Manifold learning in physical systems
Astronomical data mining and exoplanet characterization
