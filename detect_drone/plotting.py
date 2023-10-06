import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detect_drone.outputs import save_visualizations
from detect_drone.settings import OUTPUT_DIR
import seaborn as sns


def create_trajectory_plot(frame_numbers, x_coordinates, y_coordinates):
    figure, axis = plt.subplots(figsize=(5, 4), dpi=100)
    axis.plot(frame_numbers, x_coordinates, label='X Coordinate')
    axis.plot(frame_numbers, y_coordinates, label='Y Coordinate')
    axis.legend()
    return figure, axis


def draw_trajectory_plot(frame_numbers, x_coordinates, y_coordinates, current_frame):
    figure, axis = create_trajectory_plot(frame_numbers, x_coordinates, y_coordinates)
    plot_image_bgr = convert_figure_to_image(figure)
    combined_image = combine_current_frame_with_plot(current_frame, plot_image_bgr)
    save_visualizations(figure, combined_image)
    return combined_image


def convert_figure_to_image(figure):
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    plot_image_data = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
    plot_image_data = plot_image_data.reshape(int(renderer.width), int(renderer.height), -1)
    plot_image_bgr = cv2.cvtColor(plot_image_data, cv2.COLOR_RGBA2BGR)
    plot_image_bgr = plot_image_bgr[:, :, :3]
    plt.close(figure)
    return plot_image_bgr


def combine_current_frame_with_plot(current_frame, plot_image_bgr):
    frame_height, frame_width = current_frame.shape[:2]
    plot_image_height, plot_image_width = plot_image_bgr.shape[:2]
    combined_image = np.zeros((frame_height + plot_image_height, max(frame_width, plot_image_width), 3), np.uint8)
    combined_image[:frame_height, :frame_width, :3] = current_frame
    horizontal_offset = (frame_width - plot_image_width) // 2
    combined_image[frame_height:frame_height + plot_image_height,
    horizontal_offset:horizontal_offset + plot_image_width, :3] = plot_image_bgr
    return combined_image


def create_trajectory_plot_seaborn(frame_numbers, x_coordinates, y_coordinates):
    sns.set(style="darkgrid")
    plt.figure(figsize=(5, 4), dpi=100)
    sns.lineplot(x=frame_numbers, y=x_coordinates, label='X Coordinate')
    sns.lineplot(x=frame_numbers, y=y_coordinates, label='Y Coordinate')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/plot_seaborn.png')  # Save the plot directly
    plt.close()

