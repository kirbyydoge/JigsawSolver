import findcorners as fc
import cv2 as cv
import numpy as np

def get_sides(points):
    points = fc.sort_clockwise(points)
    return [[points[i], points[(i + 1) % len(points)]] for i in range(len(points))]

def combine_vertical(tile_up, corner_up, tile_down, corner_down):
    shape_up = tile_up.shape
    shape_down = tile_down.shape
    combined_corner = corner_up[0] - shape_up[0] // 2
    combined_offset = combined_corner - corner_down[0]
    shape_cmb = ((combined_offset + shape_down[0] // 2), shape_up[1])
    combined = np.zeros(shape_cmb, dtype=np.uint8)
    combined_row = 0
    for row in range(shape_up[0] // 2, shape_up[0]):
        for col in range(shape_up[1]):
            combined[combined_row, col] = tile_up[row, col] // 2
        combined_row += 1
    for row in range(0, shape_down[0] // 2):
        for col in range(shape_up[1]):
            combined[combined_offset + row, col] += tile_down[row, col] // 2
    print((combined > 120).sum())
    cv.imshow("Combined", combined)
    cv.waitKey(0)

if __name__ == "__main__":
    img = cv.imread("./res/test0_sample0.png")
    tiles, tile_corners = fc.get_corners(img)
    tile_sides = []
    for i in range(len(tiles)):
        tile = cv.cvtColor(tiles[i], cv.COLOR_GRAY2BGR)
        corners = tile_corners[i]
        sides = get_sides(corners)
        for side in sides:
            cv.line(tile, (side[0][1], side[0][0]), (side[1][1], side[1][0]), (255, 0, 0), 2)
        cv.imshow("Tile", tile)
        cv.waitKey(0)
        tile_sides.append(sides)
    # INITIAL TESTING
    tile_up = tiles[1]
    corner_up = tile_sides[1][2][1]
    tile_down = tiles[2]
    corner_down = tile_sides[2][0][0]
    combine_vertical(tile_up, corner_up, tile_down, corner_down)