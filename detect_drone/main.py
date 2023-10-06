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


def draw_trajectory_plot(frame_numbers, x_coords, y_coords, current):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.plot(frame_numbers, x_coords, label='X Coordinate')
    ax.plot(frame_numbers, y_coords, label='Y Coordinate')
    ax.legend()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    plot_img = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(int(renderer.width), int(renderer.height), -1)  # Convert to int
    plot_img_bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    plot_img_bgr = plot_img_bgr[:, :, :3]
    cv2.imshow('ss', plot_img)
    h1, w1 = current.shape[:2]
    h2, w2 = plot_img_bgr.shape[:2]

    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis[:h1, :w1, :3] = current
    offset = (w1 - w2) // 2  # Calculate the horizontal offset to center plot_img_bgr
    vis[h1:h1 + h2, offset:offset + w2, :3] = plot_img_bgr
    print(f'plot_img_bgr dimensions: {plot_img_bgr.shape}')
    print(f'current dimensions: {current.shape}')
    print(f'vis dimensions: {vis.shape}')

    fig.savefig('plot.png')
    plt.close(fig)
    cv2.imwrite('vis.png', vis)
    return vis


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
