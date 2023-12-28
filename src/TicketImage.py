from PIL import Image
import numpy as np
import cv2 as cv


class TicketImage:
    """
    Class to handle each ticket and its related data
    """
    def __init__(self, path: str, image: Image):
        self.path = path

        if image.mode == "L":
            self.grayscale_image = np.array(image)
            self.original_image = self.grayscale_image
        else:
            numpy_image = np.array(image)

            self.grayscale_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2GRAY)
            self.original_image = numpy_image
