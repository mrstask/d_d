import csv
import os

import cv2

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np

from detect_drone.filters import is_contour_large_enough
from detect_drone.helpers import key_function
from detect_drone.image_preprocessing import prepare_images

THRESHOLD_VALUE = 30
MAX_VALUE = 255
IMAGE_EXTENSION = ".jpg"
MIN_CONTOUR_AREA = 200


def detect_changes(background_gray, current_gray):
    diff = cv2.absdiff(current_gray, background_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def identify_and_show_changes(image_dir, background_filename, current_filename, min_contour_area=MIN_CONTOUR_AREA):
    background, current, background_gray, current_gray = prepare_images(image_dir, background_filename,
                                                                        current_filename)
    contours = detect_changes(background_gray, current_gray)
    tracker = None
    bbox = None
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        if is_contour_large_enough(contour, min_contour_area):
            x, y, w, h = bounding_rect
            cv2.rectangle(current, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tracker = cv2.TrackerKCF_create()
            tracker.init(current, (x, y, w, h))
            bbox = (x, y, w, h)
            break

    return tracker, bbox  # Return both tracker and bounding box


def load_frame(image_dir, filename):
    frame_path = os.path.join(image_dir, filename)
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, frame_gray


def detect_drone(background_gray, current_gray, min_contour_area=MIN_CONTOUR_AREA):
    contours = detect_changes(background_gray, current_gray)
    for contour in contours:
        if is_contour_large_enough(contour, min_contour_area):
            return cv2.boundingRect(contour)
    return None


def initialize_tracker(current, bbox):
    tracker = cv2.TrackerKCF_create()
    tracker.init(current, bbox)
    return tracker


def update_tracker(tracker, current):
    success, box = tracker.update(current)
    if success:
        bbox = tuple(int(v) for v in box)
        cv2.rectangle(current, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        return bbox, True
    return None, False


def redetect_drone(background_gray, current_gray, current):
    bbox = detect_drone(background_gray, current_gray)
    if bbox is not None:
        cv2.rectangle(current, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        return bbox
    return None


def record_data(frame_idx, bbox, trajectory_data):
    trajectory_data.append([frame_idx, *bbox])


def write_data_to_csv(trajectory_data):
    with open('trajectory_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "X", "Y", "Width", "Height"])
        writer.writerows(trajectory_data)


def create_trajectory_plot(frame_numbers, x_coordinates, y_coordinates):
    figure, axis = plt.subplots(figsize=(5, 4), dpi=100)
    axis.plot(frame_numbers, x_coordinates, label='X Coordinate')
    axis.plot(frame_numbers, y_coordinates, label='Y Coordinate')
    axis.legend()
    return figure, axis


def convert_figure_to_image(figure):
    canvas = agg.FigureCanvasAgg(figure)
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


def save_visualizations(figure, combined_image):
    figure.savefig('trajectory_plot.png')
    cv2.imwrite('combined_visualization.png', combined_image)


def draw_trajectory_plot(frame_numbers, x_coordinates, y_coordinates, current_frame):
    figure, axis = create_trajectory_plot(frame_numbers, x_coordinates, y_coordinates)
    plot_image_bgr = convert_figure_to_image(figure)
    combined_image = combine_current_frame_with_plot(current_frame, plot_image_bgr)
    save_visualizations(figure, combined_image)
    return combined_image


def compare_frames_with_background(frames_dir: str, image_extension: str = ".jpg") -> None:
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(image_extension)], key=key_function)
    if not frames:
        raise FileNotFoundError(f"No images found in the directory: {frames_dir}")

    background_filename = frames.pop(0)
    background, background_gray = load_frame(frames_dir, background_filename)

    tracker = None
    trajectory_data = []
    frame_numbers, x_coords, y_coords = [], [], []

    for idx, frame in enumerate(frames):
        print(f"Processing frame: {frame}")
        current, current_gray = load_frame(frames_dir, frame)

        if tracker is None:
            print("Initializing tracker...")
            tracker, bbox = identify_and_show_changes(frames_dir, background_filename, frame)
        else:
            print("Updating tracker...")
            bbox, success = update_tracker(tracker, current)
            if not success:
                print("Tracker update failed. Re-detecting...")
                bbox = redetect_drone(background_gray, current_gray, current)
                if bbox is not None:
                    tracker = initialize_tracker(current, bbox)

        if bbox is not None:
            record_data(idx, bbox, trajectory_data)
            frame_numbers.append(idx)
            x_coords.append(bbox[0] + bbox[2] / 2)
            y_coords.append(bbox[1] + bbox[3] / 2)

        vis = draw_trajectory_plot(frame_numbers, x_coords, y_coords, current)

        cv2.imshow("Current Image", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    write_data_to_csv(trajectory_data)


if __name__ == '__main__':
    compare_frames_with_background("frames")
