import cv2


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


def record_data(frame_idx, bbox, trajectory_data):
    trajectory_data.append([frame_idx, *bbox])
