import cv2
import numpy as np
import os

from detect_drone.helpers import key_function


def draw_trajectories(prev_pts, next_pts, frame, path_length=10):
    for i in range(len(prev_pts)):
        for j in range(1, path_length):
            if j >= len(prev_pts[i]):
                break
            cv2.line(frame, tuple(prev_pts[i][j - 1]), tuple(prev_pts[i][j]), (0, 255, 0), 2)


def optical_flow_trajectory(prev_frame, next_frame, prev_pts):
    # Ensure frames are in grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate sparse optical flow (Lucas-Kanade method)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, next_frame_gray, prev_pts, None)

    # Filter only the valid points
    mask = status.flatten() == 1
    prev_pts = prev_pts[mask]
    next_pts = next_pts[mask]

    return prev_pts, next_pts


frames_dir = 'frames'
image_extension = '.jpg'

frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(image_extension)], key=key_function)

# Example: Use the first frame to determine the dimensions
example_frame = cv2.imread(f"{frames_dir}/{frames[0]}")
height, width = example_frame.shape[:2]

# Define step size and number of frames to process
step = 20  # Example: space points 20 pixels apart
num_frames = len(frames)  # Process all available frames

for frame in frames:
    prev_frame = cv2.imread(f"{frames_dir}/{frame}")
    prev_pts = np.float32([[x, y] for y, x in np.mgrid[0:height:step, 0:width:step].reshape(2, -1).T])

    trajectories = [[] for _ in range(len(prev_pts))]

    for i in range(2, num_frames):
        next_frame = cv2.imread(f"frames/{i}.jpg")

        prev_pts, next_pts = optical_flow_trajectory(prev_frame, next_frame, prev_pts)

        for j, (new, old) in enumerate(zip(next_pts, prev_pts)):
            # Debug print to check shapes and values
            print(f"old: {old}, shape: {old.shape}, new: {new}, shape: {new.shape}")

            # Extract coordinates based on shape
            if old.shape == (2,):
                old_x, old_y = int(old[0]), int(old[1])
            elif old.shape == (1, 2):
                old_x, old_y = int(old[0, 0]), int(old[0, 1])
            else:
                print("Unexpected shape encountered, skipping this iteration.")
                continue

            # Append to trajectories
            trajectories[j].append((old_x, old_y))

        draw_trajectories(trajectories, next_pts, next_frame)

        prev_frame = next_frame.copy()
        prev_pts = next_pts.reshape(-1, 1, 2)

        cv2.imshow("Trajectories", next_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
