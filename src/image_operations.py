import os
from pathlib import Path
from PIL import Image
import cv2 as cv

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


def normalize_image(ticket_image: TicketImage, max_width) -> TicketImage:
    image = ticket_image.image

    image_ratio = image.shape[0] / image.shape[1]
    ticket_image.image = cv.resize(image, (max_width, int(max_width * image_ratio)))

    return ticket_image


def generate_scale_pyramid(image: TicketImage, scales: [float]) -> []:
    images = [None] * len(scales)
    for i, scale in enumerate(scales):
        rescaled_image = cv.resize(image.image, None, fx=scale, fy=scale)

        images[i] = TicketImage(image.path, rescaled_image)

    return images


def draw_detection(image, logo, max_loc):
    h, w = logo.shape[:2]
    cv.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 0, 0))
