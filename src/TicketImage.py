import numpy as np
import cv2 as cv


class TicketImage:
    def __init__(self, path: str, image):
        self.path = path

        numpy_image = np.array(image)
        self.image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)

    def show(self, title: str):
        cv.imshow(title, self.image)

    def get_path(self):
        return self.path

