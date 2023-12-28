import os.path
from pathlib import Path
import cv2 as cv
import pandas as pd


def save_classified_images(logos, labeled_images):
    output = "output"

    for logo in logos:
        path = os.path.join(output, logos[logo])
        path = Path(path).as_posix()
        # Create the directory for the logo
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    # Save each image in their label directory
    for image_path, predicted_label, ticket_image in labeled_images:
        ipath = ticket_image.path.split("/")[-1]
        ipath = os.path.join(output, predicted_label, ipath)
        ipath = Path(ipath).as_posix()

        cv.imwrite(ipath, cv.cvtColor(ticket_image.original_image, cv.COLOR_RGB2BGR))


def get_logo_label(logo_path: str, all_labels: dict) -> str:
    if logo_path in all_labels:
        return all_labels[logo_path]

    return "Unknown"


def load_csv(path: str) -> []:
    df = pd.read_csv(path, sep=",")
    data = dict()

    for index, row in df.iterrows():
        path = Path(row["Path"]).as_posix()
        label = row["Label"]

        data[path] = label

    return data
