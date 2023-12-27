import os.path
from pathlib import Path
import cv2


def save_classified_images(logos, images, labeled_images):
    print(labeled_images)
    print(images[0].path)
    output = "output"
    for logo in logos:
        path = os.path.join(output, logos[logo])
        path = Path(path).as_posix()
        if not os.path.exists(path):
            os.makedirs(path)
        for image in images:
            for labeled in labeled_images:
                if image.path == labeled[0] and logos[logo] == labeled[1]:
                    ipath = image.path.split("/")[-1]
                    ipath = os.path.join(path, ipath)
                    ipath = Path(ipath).as_posix()
                    cv2.imwrite(ipath, image.original_image)



