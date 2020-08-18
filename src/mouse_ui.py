# pylint: disable=E1126

import functools
import cv2 as cv
import pathlib
import math
from typing import Optional, List, Tuple
import numpy as np

Point = Tuple[int, int]

COLOR_NORMAL = (255, 102, 144)
COLOR_SELECTED = (102, 102, 255)
ALPHA = 0.4

circles: List[Point] = []
idx_active: Optional[int] = None

def close_enough(p1: Point, p2: Point, dist=15):
    x1, y1 = p1
    x2, y2 = p2
    res = math.hypot(x1 - x2, y1 - y2) < dist
    return res


def redraw():
    global img
    img = img_orig.copy()
    if len(circles) == 4:
        overlay = img.copy()
        overlay = cv.fillConvexPoly(overlay, np.array(circles, dtype=np.int32), (255, 255, 255))
        img = cv.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0, img)

    for i, pos in enumerate(circles):
        if i == idx_active:
            img = cv.circle(img, pos, 10, COLOR_SELECTED, -1)
        else:
            img = cv.circle(img, pos, 10, COLOR_NORMAL, -1)


def sort_circles():
    global circles
    assert len(circles) >= 4
    points = np.asarray(circles[-4:])
    key = functools.cmp_to_key(np.cross)
    vecs = points[1:] - points[0]
    xs = np.apply_along_axis(key, 1, vecs)
    indices = np.argsort(xs)
    indices = [0] + list(1 + indices)
    circles = [tuple(p) for p in points[indices]]


def edit_circle(event, x, y, flags, param):
    global idx_active
    if event == cv.EVENT_LBUTTONDBLCLK:
        idx = next((i for i, circle in enumerate(circles) if close_enough(circle, (x, y))), None)
        if idx is None:
            # add circle
            circles.append((x, y))
            if len(circles) >= 4:
                sort_circles()
        else:
            # delete circle
            del circles[idx]
        redraw()

    elif event == cv.EVENT_LBUTTONDOWN:
        idx_active = next((i for i, circle in enumerate(circles) if close_enough(circle, (x, y))), None)
        redraw()

    elif (idx_active is not None) and event == cv.EVENT_MOUSEMOVE:
        circles[idx_active] = (x, y)
        redraw()

    elif (idx_active is not None) and event == cv.EVENT_LBUTTONUP:
        circles[idx_active] = (x, y)
        idx_active = None
        if len(circles) == 4:
            sort_circles()
        redraw()


p = pathlib.Path(".") / "lenna.png"
assert p.exists()
img_orig = cv.imread(p.as_posix())
img = img_orig.copy()
cv.namedWindow("image")
cv.setMouseCallback("image", edit_circle)
while 1:
    cv.imshow("image", img)
    if cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
