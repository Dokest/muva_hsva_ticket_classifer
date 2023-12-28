import numpy as np
from matplotlib import pyplot as plt
from src.colors import color_ok, color_fail


def logo_score(logos, predicted_label, gt):
    accuracy = []
    a_logos = []
    for j, logo in enumerate(logos):
        total_score = 0
        a_logos.append(logos[logo])
        for i in range(len(predicted_label)):
            path, predicted, _ = predicted_label[i]
            # Check the label
            if predicted == logos[logo]:
                color_fn = color_ok if predicted == gt[path] else color_fail

                print(color_fn(f"Predicted: {predicted}, Real: {gt[path]} :: {path}"))
                # Check the real label
                if predicted == gt[path]:
                    total_score += 1
        accuracy.append(total_score)
    errors = [None] * len(accuracy)

    # Normalize the results
    for i in range(len(accuracy)):
        errors[i] = 5 - accuracy[i]

    accuracy = np.array(accuracy) / 5 * 100
    errors = np.array(errors) / 5 * 100
    display_score(accuracy, errors, a_logos)


def display_score(accuracy, errors, logos):
    ca = (93/255, 156/255, 89/255)
    ce = (223/255, 46/255, 56/255)
    bar_width = 0.5

    # Create the figure
    fig, ax = plt.subplots()

    # Create stacked bars
    ax.bar(logos, accuracy, label='Acierto', color=ca, width=bar_width)
    ax.bar(logos, errors, bottom=accuracy, label='Error', color=ce, width=bar_width)

    ax.legend()
    ax.set_xlabel('Categorías')
    ax.set_ylabel('Porcentaje')
    ax.set_title('Aciertos por clase')
    plt.show()


def display_total_score(predicted_labels, gt_labels):
    total_score = 0
    for i in range(len(predicted_labels)):
        path, predicted, _ = predicted_labels[i]

        color_fn = color_ok if predicted == gt_labels[path] else color_fail

        print(color_fn(f"Predicted: {predicted}, Real: {gt_labels[path]} :: {path}"))
        # Check the real label
        if predicted == gt_labels[path]:
            total_score += 1
    # Normalize the results
    total_score = [total_score / len(predicted_labels) * 100]
    error = [100 - total_score[0]]

    print(f"Accuracy: {total_score}%")
    display_score(total_score, error, ['Total'])
