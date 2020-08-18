# pylint: disable=E1126

import pathlib
import functools
from typing import Optional, List, Tuple, Iterable
import numpy as np
import cv2 as cv


Point2D = Tuple[int, int]

COLOR_NORMAL = (255, 102, 144)
COLOR_SELECTED = (102, 102, 255)
ALPHA = 0.4

CLOSE_ENOUGH_DIST = 15

# points stores (x, y) points for rectangular vertices
points: List[Point2D] = []
# idx_active is the index of points currently selected
idx_active: Optional[int] = None


def close_enough(p1: Point2D, p2: Point2D):
    x1, y1 = p1
    x2, y2 = p2
    res = np.hypot(x1 - x2, y1 - y2) < CLOSE_ENOUGH_DIST
    return res


def current_img_mat():
    """Returns a image matrix by superposing IMG_ORIG with circles and polygons
    specified by the global variable `points`.

    """
    im = IMG_ORIG.copy()

    if len(points) == 4:
        overlay = im.copy()
        overlay = cv.fillConvexPoly(
            overlay, np.array(points, dtype=np.int32), (255, 255, 255)
        )
        im = cv.addWeighted(overlay, ALPHA, im, 1 - ALPHA, 0, im)

    for i, pos in enumerate(points):
        if i == idx_active:
            im = cv.circle(im, pos, 10, COLOR_SELECTED, -1)
        else:
            im = cv.circle(im, pos, 10, COLOR_NORMAL, -1)
    return im


def sort_points(pts: List[Point2D]) -> List[Point2D]:
    """Sort points in clockwise order such that
    [top_left, top_right, bottom_right, bottom_left]
    """
    assert len(pts) >= 4
    pts = np.asarray(pts[-4:])

    # Leaving the first point. Sort the rest in clockwise order.
    key = functools.cmp_to_key(lambda p1, p2: -np.cross(p1, p2))
    vecs = pts[1:] - pts[0]
    xs = np.apply_along_axis(key, 1, vecs)
    indices = np.argsort(xs)
    indices = [0] + list(1 + indices)
    pts = [tuple(p) for p in pts[indices]]

    # Top-left point comes first.
    idx_tl = min(range(4), key=lambda i: sum(pts[i]))
    return pts[idx_tl:] + pts[:idx_tl]


def edit_circle(event, x, y, flags, param):
    """Callback function triggered by mouse events

    It updates following global varibles
    - points (List[Point2D])
    - idx_active
    - img

    """
    global points
    global idx_active
    global img

    if event == cv.EVENT_LBUTTONDBLCLK:
        idx = next(
            (i for i, circle in enumerate(points) if close_enough(circle, (x, y))), None
        )
        if idx is None:
            # add point
            points.append((x, y))
            if len(points) >= 4:
                points = sort_points(points)
        else:
            # delete point
            del points[idx]

    elif event == cv.EVENT_LBUTTONDOWN:
        idx_active = next(
            (i for i, circle in enumerate(points) if close_enough(circle, (x, y))), None
        )

    elif (idx_active is not None) and event == cv.EVENT_MOUSEMOVE:
        points[idx_active] = (x, y)

    elif (idx_active is not None) and event == cv.EVENT_LBUTTONUP:
        points[idx_active] = (x, y)
        idx_active = None
        if len(points) == 4:
            points = sort_points(points)

    else:
        # no image update
        return

    img = current_img_mat()


p = pathlib.Path(".") / "lenna.png"
assert p.exists()
IMG_ORIG = cv.imread(p.as_posix())
img = IMG_ORIG.copy()
cv.namedWindow("image")
cv.setMouseCallback("image", edit_circle)
while 1:
    cv.imshow("image", img)
    if cv.waitKey(1) == ord("q"):
        break
cv.destroyAllWindows()
