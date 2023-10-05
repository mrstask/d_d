import cv2


def is_contour_large_enough(contour, min_contour_area: int) -> bool:
    return cv2.contourArea(contour) >= min_contour_area
