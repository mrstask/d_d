import csv

import cv2

from detect_drone.settings import OUTPUT_DIR


def write_data_to_csv(trajectory_data):
    with open(f'{OUTPUT_DIR}/trajectory_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Width", "Height"])
        writer.writerows(trajectory_data)


def save_visualizations(figure, combined_image):
    figure.savefig(f'{OUTPUT_DIR}/trajectory_plot.png')
    cv2.imwrite(f'{OUTPUT_DIR}/combined_visualization.png', combined_image)
