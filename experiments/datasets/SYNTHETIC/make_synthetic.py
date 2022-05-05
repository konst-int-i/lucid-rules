from sklearn.datasets import make_classification


def main():
    n_features = 20
    feature_names = [f"feat_{i+1}" for i in range(n_features)]

    synthetic_np = make_classification(
        n_features=n_features,
        n_informative=4,
        n_redundant=4,
        n_repeated=2,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=1
    )
    return synthetic_np

if __name__ == "__main__":
    synthetic_np = main()
    # synthetic_np.to_csv("experiments/datasets/SYNTHETIC/data.csv")