import os
import multiprocessing
import numpy as np
from src.TicketImage import TicketImage
from src.classify import classify
from src.image_operations import load_images, generate_scale_pyramid, general_preprocess
from src.scoring import logo_score, display_total_score
from src.utils import save_classified_images, load_logos_csv, get_logo_label


def main():
    logo_labels = load_logos_csv("etiquetas/logos.csv")
    ground_truth_labels = load_logos_csv("etiquetas/gt.csv")

    logo_folder = os.path.join("images", "logos")
    image_folder = "images"

    # Load images
    images = load_images(image_folder)
    images = [general_preprocess(image, 1000) for image in images]

    # Load and normalize logos
    logos = load_images(logo_folder)
    logos = [general_preprocess(logo, 100) for logo in logos]

    # Generate multiple logo version with different scales
    scales = [1.0, 1.25, 1.5, 2, 3, 5]
    logos = np.array([generate_scale_pyramid(logo, scales) for logo in logos]).flatten()

    labeled_images: [(str, str, TicketImage)] = []

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Execute in the classification in parallel
        image_logo_pairs = [(image, logos) for image in images]
        results = pool.starmap(classify, image_logo_pairs)

        # Get results for each image
        for i, (conf, logo_index, image_path) in enumerate(results):
            logo_path = logos[logo_index].path
            predicted_label = get_logo_label(logo_path, logo_labels)

            labeled_images.append((image_path, predicted_label, images[i]))

    save_classified_images(logo_labels, labeled_images)
    logo_score(logo_labels, labeled_images, ground_truth_labels)
    display_total_score(labeled_images, ground_truth_labels)


if __name__ == "__main__":
    main()
