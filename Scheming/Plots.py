import matplotlib.pyplot as plt
from feature_importance import get_fnr_results
from active_learning import get_accuracies

def plot_learning_curve(accuracies):
    plt.figure()
    plt.plot(accuracies, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Performance")
    plt.show()


def plot_fnr(fnr_results):
    features = [x[0] for x in fnr_results]
    values = [x[1] for x in fnr_results]
    plt.figure()
    plt.barh(features, values)
    plt.xlabel("FNR Increase (Permutation)")
    plt.title("Feature Influence (FNR Proxy)")
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    plot_learning_curve(get_accuracies())
    plot_fnr(get_fnr_results())