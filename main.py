import os
import pandas as pd
from pathlib import Path
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt

from src.classify import classify
from src.colors import color_ok, color_fail
from src.image_operations import load_images, generate_scale_pyramid, general_preprocess


def get_logo_label(logo_path: str, all_labels: dict) -> str:
    if logo_path in all_labels:
        return all_labels[logo_path]

    return "Unknown"


def load_logos_csv(path: str) -> []:
    df = pd.read_csv(path, sep=",")
    data = dict()

    for index, row in df.iterrows():
        path = Path(row["Path"]).as_posix()
        label = row["Label"]

        data[path] = label

    return data


def score(predicted_labels, gt_labels):
    total_score = 0

    for i in range(len(predicted_labels)):
        path, predicted = predicted_labels[i]

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


def main():
    logo_labels = load_logos_csv("etiquetas/logos.csv")
    ground_truth_labels = load_logos_csv("etiquetas/gt.csv")

    logo_folder = os.path.join("images", "logos")
    image_folder = "images"

    # Load images
    images = load_images(image_folder)
    images = [general_preprocess(image, 1000) for image in images]

    # Load and normalize logos
    logos = load_images(logo_folder)
    logos = [general_preprocess(logo, 100) for logo in logos]

    # Generate multiple logo version with different scales
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 5]
    logos = np.array([generate_scale_pyramid(logo, scales) for logo in logos]).flatten()

    labeled_images = []

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Execute in the classification in parallel
        image_logo_pairs = [(image, logos) for image in images]
        results = pool.starmap(classify, image_logo_pairs)

        # Get results for each image
        for i, (conf, logo_index, image_path) in enumerate(results):
            logo_path = logos[logo_index].path
            predicted_label = get_logo_label(logo_path, logo_labels)

            labeled_images.append((image_path, predicted_label))
            # print(i, image_path, label)

    score(labeled_images, ground_truth_labels)


if __name__ == "__main__":
    main()
