import cv2 as cv
import numpy as np
from functools import reduce
import operator
import math
import itertools

FIND_MAX_WIDTH = 10

def angle(p0, p1, p2, acute=True):
    v0 = (p0[0] - p1[0], p0[1] - p1[1])
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    # Adding arbitrarily large epsilon value to fix floating point inaccuracy (clamp [-1, 1])
    if acute:
        return np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 0.0001)) * 180 / np.pi
    else:
        return 360 - np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 0.0001)) * 180 / np.pi

def bounding_box(points):
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)
    return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]

def bounding_box_center(points):
    bbox = bounding_box(points)
    return ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2)

def line_length(x0, y0, x1, y1):
    dif_x = x1 - x0
    dif_y = y1 - y0
    return math.sqrt(dif_x * dif_x + dif_y * dif_y)

def rect_area(sides, angles):
    area1 = 0.5 * sides[0] * sides[1] * math.sin(math.radians(angles[1]))
    area2 = 0.5 * sides[2] * sides[3] * math.sin(math.radians(angles[3]))
    return area1 + area2

def sort_clockwise(points):
    # center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    # return sorted(points, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    points = np.array(points)
    cx, cy = points.mean(0)
    x, y = points.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return points[indices]

def rect_score(points):
    points = sort_clockwise(points)
    sides = [line_length(points[i][0], points[i][1] , points[(i + 1) % len(points)][0], points[(i + 1) % len(points)][1]) for i in range(len(points))]
    angles = [angle(points[(i - 1) % len(points)], points[(i)], points[(i + 1) % len(points)]) for i in range(len(points))]
    # multiplier = 1
    # for ang in angles:
    #     multiplier -= abs(90 - ang) * 0.005
    return rect_area(sides, angles) # * multiplier

def pick_local_maxima(points, corner_scores, image_shape, window_width):
    local_maxima = set()
    for i, j in points:
        cur_max = -np.inf
        max_i = -1
        max_j = -1
        for row in range(-window_width, window_width + 1):
            row_idx = i + row
            if row_idx < 0 or row_idx >= image_shape[0]:
                continue
            for col in range(-window_width, window_width + 1):
                col_idx = j + col
                if col_idx < 0 or col_idx >= image_shape[1]:
                    continue
                if corner_scores[row_idx, col_idx] > cur_max:
                    max_i = row_idx
                    max_j = col_idx
                    cur_max = corner_scores[row_idx, col_idx]
        if cur_max > 0:
            local_maxima.add((max_i, max_j))
    return local_maxima

def get_tiles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY_INV)
    # thresh = cv.blur(thresh, ksize=(3, 3))

    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    height, width = thresh.shape[:2]
    vis = np.zeros((height, width, 3), np.uint8)

    cv.drawContours(vis, contours, -1, (128,255,255), -1)

    # cv.imshow("img", img)
    # cv.imshow("grey", gray)
    # cv.imshow("thresh", thresh)
    # cv.imshow("contour", vis)
    # cv.waitKey(0)

    # Parse pieces into tiles such that each tile only has 1 piece
    tiles = []
    color_tiles = []
    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i]) 
        if w < 10 and h < 10:
            continue
        shape, tile, color_tile = np.zeros(thresh.shape[:2]), np.zeros((300,300), 'uint8'), np.zeros((300,300,3), 'uint8') 
        cv.drawContours(shape, [contours[i]], -1, 1, -1)
        color_shape = np.zeros_like(img)
        color_shape[shape == 1] = img[shape == 1]
        shape = (vis[:,:,1] * shape[:,:])[y:y+h, x:x+w]
        color_shape = color_shape[y:y+h, x:x+w]
        tile[(300-h)//2:(300-h)//2+h , (300-w)//2:(300-w)//2+w] = shape
        color_tile[(300-h)//2:(300-h)//2+h , (300-w)//2:(300-w)//2+w] = color_shape
        tiles.append(tile)
        color_tiles.append(color_tile)
    return tiles, color_tiles

def get_corners(img):
    tiles, color_tiles = get_tiles(img)

    tile_corners = []
    for idx, image in enumerate(tiles):
        dst = cv.cornerHarris(image, 7, 5, 0.07)
        color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        corner_coords = []
        for i, row in enumerate(dst):
            for j, _ in enumerate(row):
                if (dst[i, j] > 0.05 * dst.max()):
                    corner_coords.append((i, j))
                    # cv.circle(color, (j, i), 2, (0, 255, 0), -1)
        # cv.imshow("Harris Corner", color)
        # cv.waitKey(0)
        # print(f"Num harris corners: {len(corner_coords)}")

        # Reduce search space by only picking maximal values in a centered window space
        local_max_coords = pick_local_maxima(corner_coords, dst, image.shape, FIND_MAX_WIDTH)
        # for coord in local_max_coords:
        #     cv.circle(color, (coord[1], coord[0]), 10, (0, 255, 0), -1)
        # cv.imshow("Local Max", color)
        # cv.waitKey(0)
        # print(f"Num local maxima corners: {len(local_max_coords)}")

        max_list = []
        center = bounding_box_center(local_max_coords)
        pivot = (center[0], center[1] + 10)
        # cv.circle(color, (int(center[1]), int(center[0])), 5, (0, 255, 0), -1)
        # cv.circle(color, (int(pivot[1]), int(pivot[0])), 5, (255, 0, 0), -1)

        # Heuristic: Divide candidate points into 8 subgroups
        groups = [[] for i in range(8)]
        for coord in local_max_coords:
            ang = angle(coord, center, pivot, coord[0] <= center[0])
            groups[int(ang // (360 // 8))].append(coord)

        # Heuristic: Never pick 2 points from the same quadrant
        iterator = iter([])
        for prod in itertools.product([0, 1], [2, 3], [4, 5], [6, 7]):
            iterator = itertools.chain(iterator, itertools.product(*[groups[idx] for idx in prod]))

        # View computation efficiency 
        # print(sum(1 for _ in itertools.combinations(corner_coords, 4))) # All possible initial quadruples
        # print(sum(1 for _ in itertools.combinations(local_max_coords, 4))) # All possible reduced quadruples
        # print(sum(1 for _ in iterator)) # 1 corner from each (matchable) group

        # Recompute iterators as we exhausted it by counting the number of elements above
        iterator = iter([])
        for prod in itertools.product([0, 1], [2, 3], [4, 5], [6, 7]):
            iterator = itertools.chain(iterator, itertools.product(*[groups[idx] for idx in prod]))

        # Find max scoring quadruple
        cur_max = -np.inf
        max_list = []
        for subset in iterator:
            score = rect_score(subset)
            if score > cur_max:
                cur_max = score
                max_list = subset
        tile_corners.append(max_list)
    return tiles, tile_corners, color_tiles

if __name__ == "__main__":
    img = cv.imread("./res/test0.png")
    tiles, tile_corners, color_tiles = get_corners(img)
    for i in range(len(tiles)):
        # tile = cv.cvtColor(tiles[i], cv.COLOR_GRAY2BGR)
        tile = color_tiles[i]
        corners = tile_corners[i]
        for point in corners:
            cv.circle(tile, (point[1], point[0]), 10, (0, 0, 255), -1)
        cv.imshow(f"Tile{i}", tile)
    cv.waitKey(0)