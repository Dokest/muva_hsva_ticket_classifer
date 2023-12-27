from src.TicketImage import TicketImage
import cv2 as cv
from matplotlib import pyplot as plt

from src.image_operations import draw_detection


def classify(image: TicketImage, logos: [TicketImage]) -> (float, int, float):
    max_conf: float = 0
    max_conf_index: int = 0
    max_conf_loc = (0, 0)

    image_data = image.grayscale_image
    image_path = image.path

    for i, logo in enumerate(logos):
        conf, max_loc = detect_logo_confidence(image_data, logo.grayscale_image)

        if conf > max_conf:
            max_conf = conf
            max_conf_index = i
            max_conf_loc = max_loc

    # Draw the detection for the best case
    draw_detection(image.original_image, logos[max_conf_index].grayscale_image, max_conf_loc)

    # Draw the image with its best match
    plt.subplot(121)
    plt.imshow(image.original_image)
    plt.subplot(122)
    plt.imshow(logos[max_conf_index].original_image)
    plt.title("Confidence: {:.2f}".format(max_conf))
    plt.show()

    return max_conf, max_conf_index, image_path


def detect_logo_confidence(image, logo) -> (float, (int, int)):
    result = cv.matchTemplate(image, logo, cv.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv.minMaxLoc(result)

    return maxVal, maxLoc
