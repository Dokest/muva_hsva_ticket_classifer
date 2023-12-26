import os
import cv2 as cv
import numba
from PIL import Image
from matplotlib import pyplot
import pandas as pd
from pathlib import Path

from src.TicketImage import TicketImage


def load_images(folder: str) -> [TicketImage]:
    image_paths = os.listdir(folder)

    images = []

    for image_path in image_paths:
        complete_path = os.path.join(folder, image_path)

        if not os.path.isfile(complete_path):
            continue

        image = Image.open(complete_path)

        if image.mode in ('P', 'LA', 'RGBA'):
            image = image.convert('RGBA')

        image = image.convert("L")

        images.append(TicketImage(Path(complete_path).as_posix(), image))

    return images


def detect_logo_confidence(image: numba.uint8[:, :], logo: numba.uint8[:, :]) -> float:
    result = cv.matchTemplate(image, logo, cv.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv.minMaxLoc(result)

    # def draw_detection():
    #     print(logo.shape)
    #     h, w, _ = logo.shape
    #     cv.rectangle(image, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), (255, 0, 0))

    # draw_detection()
    # cv.imshow("ASDASD", image)
    # cv.waitKey(0)

    return maxVal


def classify(image, logos: [TicketImage]) -> (float, int):
    max_conf: float = 0
    max_conf_index: int = 0

    for i, logo in enumerate(logos):
        conf = detect_logo_confidence(image, logo.image)

        if conf > max_conf:
            max_conf = conf
            max_conf_index = i

    return max_conf, max_conf_index


@numba.jit(nopython=True)
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

    images = load_images(image_folder)
    logos = load_images(logo_folder)
    
    for i in range(len(images)):
        conf, max_index = classify(images[i].image, logos)

        logo_path = logos[max_index].path
        label = get_logo_label(logo_path, logo_labels)

        print(label)
        labels = []
        labels.append(label)
    total_score = score(labels, image_labels)


if __name__ == "__main__":
    main()
