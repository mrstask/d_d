import os

import cv2


def extract_frames_from_video(video_path: str, output_dir: str = "frames") -> None:
    """
    Extract frames from a video and save them as images in a specified directory.
    Parameters:
        video_path (str): The file path of the input video.
        output_dir (str, optional): The directory where extracted frames will be saved.
                                    Defaults to "frames".
    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(os.path.join(output_dir, f"{frame_idx}.jpg"), frame)
        frame_idx += 1

    cap.release()


def key_function(string):
    return int(string.split(".")[0])
