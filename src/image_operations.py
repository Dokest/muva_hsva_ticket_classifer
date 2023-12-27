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

        if image.mode in ("P", "LA", "RGBA", "L"):
            image = image.convert("RGB")

        images.append(TicketImage(Path(complete_path).as_posix(), image))

    return images


def normalize_image(ticket_image: TicketImage, max_width) -> TicketImage:
    image = ticket_image.grayscale_image
    image_ratio = image.shape[0] / image.shape[1]

    ticket_image.grayscale_image = cv.resize(ticket_image.grayscale_image, (max_width, int(max_width * image_ratio)))
    ticket_image.original_image = cv.resize(ticket_image.original_image, (max_width, int(max_width * image_ratio)))

    return ticket_image


def generate_scale_pyramid(image: TicketImage, scales: [float]) -> []:
    images = [None] * len(scales)
    for i, scale in enumerate(scales):
        rescaled_image = cv.resize(image.grayscale_image, None, fx=scale, fy=scale)

        new_image = TicketImage(image.path, Image.fromarray(rescaled_image))
        new_image.original_image = image.original_image

        images[i] = new_image

    return images


def draw_detection(image, logo, max_loc):
    h, w = logo.shape[:2]

    cv.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 0, 0))


def general_preprocess(image: TicketImage, max_width):
    normalized_image = normalize_image(image, max_width)

    # normalized_image.grayscale_image = cv.equalizeHist(normalized_image.grayscale_image)
    # normalized_image.grayscale_image = cv.adaptiveThreshold(normalized_image.grayscale_image,255,
    #                                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    return normalized_image
