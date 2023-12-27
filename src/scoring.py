import numpy as np
from matplotlib import pyplot as plt

from src.colors import color_ok, color_fail


def display_score(predicted_labels, gt_labels):
    total_score = 0

    for i in range(len(predicted_labels)):
        path, predicted, _ = predicted_labels[i]

        color_fn = color_ok if predicted == gt_labels[path] else color_fail

        print(color_fn(f"Predicted: {predicted}, Real: {gt_labels[path]} :: {path}"))

        if predicted == gt_labels[path]:
            total_score += 1

    total_score = total_score / len(predicted_labels) * 100
    error = 100 - total_score
    print(f"Accuracy: {total_score}%")

    bars = ["Accuracy"] + ["Error"]
    colors = ["green"] + ["red"]

    plt.bar(np.arange(2), [total_score] + [error], color=colors, tick_label=bars)
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.title("Summary")
    plt.show()
