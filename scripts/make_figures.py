import matplotlib.pyplot as plt

def plot_embedding(X_emb, labels, title):
    plt.figure()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels, s=5)
    plt.title(title)
    plt.savefig(f"results/figures/{title}.png")
    plt.close()
