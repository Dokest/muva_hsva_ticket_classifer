import os
import numba
import pandas as pd
from pathlib import Path
import multiprocessing
import numpy as np

from src.classify import classify
from src.image_operations import load_images, generate_scale_pyramid, normalize_image


def get_logo_label(logo_path: str, all_labels: numba.types.DictType) -> str:
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


def score(labels, image_labels):
    total_score = 0
    for i, logo_path in range(len(labels)):
        if labels[i] == image_labels[logo_path]:
            total_score += 1

    return total_score / len(labels) * 100


def main():
    logo_labels = load_logos_csv("etiquetas/logos.csv")
    image_labels = load_logos_csv("etiquetas/gt.csv")

    logo_folder = os.path.join("images", "logos")
    image_folder = "images"

    # Load images
    images = load_images(image_folder)
    images = [normalize_image(image, 1000) for image in images]

    # Load and normalize logos
    logos = load_images(logo_folder)
    logos = [normalize_image(logo, 100) for logo in logos]

    # Generate multiple logo version with different scales
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 5]
    logos = np.array([generate_scale_pyramid(logo, scales) for logo in logos]).flatten()

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Execute in the classification in parallel
        image_logo_pairs = [(image, logos) for image in images]
        results = pool.starmap(classify, image_logo_pairs)

        # Get results for each image
        for i, (conf, logo_index) in enumerate(results):
            logo_path = logos[logo_index].path
            label = get_logo_label(logo_path, logo_labels)
            print(i, label)

    # total_score = score(labels, image_labels)


if __name__ == "__main__":
    main()
