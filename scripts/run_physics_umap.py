from src.preprocess import load_data, PHL_FEATURES
from src.physics_umap import run_physics_umap
from src.clustering import run_kmeans
from src.evaluation import evaluate_clusters

print("Loading data...")

X = load_data()

print("Running Physics-Informed UMAP...")

embedding = run_physics_umap(X, PHL_FEATURES)

labels = run_kmeans(embedding)

metrics = evaluate_clusters(embedding, labels)

print("\nPhysics-Informed UMAP Results:\n")
print(metrics)
