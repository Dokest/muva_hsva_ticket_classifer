from src.TicketImage import TicketImage
import cv2 as cv
from matplotlib import pyplot as plt

from src.image_operations import draw_detection


def classify(image: TicketImage, logos: [TicketImage]) -> (float, int, str):
    """
    Returns the data with the information of the classification.
    The first return value is the **confidence**, the second is the **logo index** and the last is the **path of the image**.
    """
    # Prepare the initial data state
    max_conf: float = 0
    max_conf_index: int = 0
    max_conf_loc = (0, 0)  # This variable is used when the show_detection function below is uncommented

    image_data = image.grayscale_image
    image_path = image.path

    # For each logo, search the image with it and then get the confidence value
    for i, logo in enumerate(logos):
        conf, max_loc = detect_logo_confidence(image_data, logo.grayscale_image)

        # Only get the logo that returns the maximum confidence
        if conf > max_conf:
            max_conf = conf
            max_conf_index = i
            max_conf_loc = max_loc

    # Uncomment to show where in the image the detection happened
    # show_detection(image.original_image, logos[max_conf_index].original_image, max_conf_loc, max_conf)

    return max_conf, max_conf_index, image_path


def detect_logo_confidence(image, logo) -> (float, (int, int)):
    result = cv.matchTemplate(image, logo, cv.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv.minMaxLoc(result)

    return maxVal, maxLoc


def show_detection(original_image, logo_original, loc, confidence):
    # Draw the detection for the best case
    draw_detection(original_image, logo_original, loc)

    # Draw the image with its best match
    plt.subplot(121)
    plt.imshow(original_image)
    plt.subplot(122)
    plt.imshow(logo_original)
    plt.title("Confidence: {:.2f}".format(confidence))
    plt.show()
