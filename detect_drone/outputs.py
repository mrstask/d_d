import csv

import cv2


def write_data_to_csv(trajectory_data):
    with open('trajectory_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Width", "Height"])
        writer.writerows(trajectory_data)


def save_visualizations(figure, combined_image):
    figure.savefig('trajectory_plot.png')
    cv2.imwrite('combined_visualization.png', combined_image)
