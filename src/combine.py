import findcorners as fc
import cv2 as cv
import numpy as np
import math
import itertools

TILE_SRC_ANCHOR = 2
TILE_TGT_ANCHOR = 0
COMBINE_CHECK_THICKNESS = 50

def get_sides(points):
    points = fc.sort_clockwise(points)
    return [[points[i], points[(i + 1) % len(points)]] for i in range(len(points))]

def combine_vertical_color(tile_src, corner_src, side_src, tile_tgt, corner_tgt, side_tgt, thickness=COMBINE_CHECK_THICKNESS):
    tile_up, corner_up = rotate(tile_src, corner_src[1], TILE_SRC_ANCHOR - side_src)
    tile_down, corner_down = rotate(tile_tgt, corner_tgt[0], TILE_TGT_ANCHOR - side_tgt)
    endpoint_up = rotate_point(corner_src[0], TILE_SRC_ANCHOR - side_src)
    shape_up = tile_up.shape
    shape_down = tile_down.shape
    combined = np.zeros((thickness * 2, endpoint_up[1] - corner_up[1], 3), dtype=np.uint8)
    for row in range(thickness * 2):
        tile_row = corner_up[0] + row - thickness
        if tile_row >= 0 and tile_row < shape_up[0]:
            for col in range(endpoint_up[1] - corner_up[1]):
                tile_col = corner_up[1] + col
                if tile_col >= 0 and tile_col < shape_up[1]:
                    combined[row, col] = tile_up[tile_row, tile_col]
    for row in range(thickness * 2):
        tile_row = corner_down[0] + row - thickness
        if tile_row >= 0 and tile_row < shape_up[0]:
            for col in range(endpoint_up[1] - corner_up[1]):
                tile_col = corner_down[1] + col
                if tile_col >= 0 and tile_col < shape_down[1]:
                    if combined[row, col].all() == 0:
                        combined[row, col] = tile_down[tile_row, tile_col]
                    elif tile_down[tile_row, tile_col].any() != 0:
                        combined[row, col] = (combined[row, col] + tile_down[tile_row, tile_col]) / 2
    return combined


def combine_vertical(tile_src, corner_src, side_src, tile_tgt, corner_tgt, side_tgt, thickness=COMBINE_CHECK_THICKNESS):
    tile_up, corner_up = rotate(tile_src, corner_src[1], TILE_SRC_ANCHOR - side_src)
    tile_down, corner_down = rotate(tile_tgt, corner_tgt[0], TILE_TGT_ANCHOR - side_tgt)
    endpoint_up = rotate_point(corner_src[0], TILE_SRC_ANCHOR - side_src, side_tgt.shape)
    shape_up = tile_up.shape
    shape_down = tile_down.shape
    combined = np.zeros((thickness * 2, endpoint_up[1] - corner_up[1]), dtype=np.uint8)
    for row in range(thickness * 2):
        tile_row = corner_up[0] + row - thickness
        if tile_row >= 0 and tile_row < shape_up[0]:
            for col in range(endpoint_up[1] - corner_up[1]):
                tile_col = corner_up[1] + col
                if tile_col >= 0 and tile_col < shape_up[1]:
                    combined[row, col] = tile_up[tile_row, tile_col] // 2
    for row in range(thickness * 2):
        tile_row = corner_down[0] + row - thickness
        if tile_row >= 0 and tile_row < shape_up[0]:
            for col in range(endpoint_up[1] - corner_up[1]):
                tile_col = corner_down[1] + col
                if tile_col >= 0 and tile_col < shape_down[1]:
                    combined[row, col] += tile_down[tile_row, tile_col] // 2
    return combined, ((combined > 128).sum() + (combined == 0).sum()) / (combined.shape[0] * combined.shape[1])

def sample_color(tile, row, col, area=3):
    sum_color = np.zeros_like(tile[0, 0])
    for i in range(area):
        row_idx = row + i - area//2
        if row_idx >= tile.shape[0] or row_idx < 0:
            continue
        for j in range(area):
            col_idx = col + j - area//2
            if col_idx >= tile.shape[1] or col_idx < 0:
                continue
            sum_color += tile[row_idx, col_idx]
    return sum_color / (3 * 255 * area * area)


def get_edge_features(tile_src, corners_src, n_parts=50):
    histogram = np.zeros(n_parts)
    color_samples = np.zeros((n_parts, 3))
    start_point = corners_src[0]
    end_point = corners_src[1]
    check_points = zip(np.linspace(start_point[0], end_point[0], n_parts), np.linspace(start_point[1], end_point[1], n_parts))
    check_vec = (end_point[1] - start_point[1], end_point[0] - start_point[0])
    control_vec = -np.array([check_vec[0], -check_vec[1]])
    try:
        control_vec = control_vec / np.linalg.norm(control_vec)
    except:
        return histogram
    for i, point in enumerate(check_points):
        max_swing = -99999
        for swing in range(-50, 50, 1):
            pos_point = control_vec * swing + point
            # pos_show = cv.circle(tile_src.copy(), (int(pos_point[1]), int(pos_point[0])), 3, (127, 127, 127), 2)
            # cv.imshow("shw", pos_show)
            # cv.waitKey(0)
            try:
                if tile_src[int(pos_point[0]), int(pos_point[1])] > 0:
                    max_swing = swing
            except:
                break
        histogram[i] = max_swing
        sample_pt = control_vec * (max_swing-5) + point
        # pos_show = cv.circle(tile_src.copy(), (int(sample_pt[1]), int(sample_pt[0])), 3, (127, 127, 127), 2)
        # cv.imshow("shw", pos_show)
        # cv.waitKey(0)
        color_samples[i] = sample_color(tile_src, int(sample_pt[0]), int(sample_pt[1]))
    return histogram, color_samples

def new_pixel(x, y, theta, img_width, img_height):
    sin = math.sin(theta)
    cos = math.cos(theta)
    x_adj = x - img_width/2
    y_adj = y - img_height/2
    x_new = x_adj * cos - y_adj * sin + img_width/2
    y_new = x_adj * sin + y_adj * cos + img_height/2
    return (int(y_new), int(x_new))

def rotate_complete(tile, sides, count):
    count = count % 4
    if count == 0:
        return tile, sides
    elif count == 1:
        rotation = cv.ROTATE_90_CLOCKWISE
    elif count == 2:
        rotation = cv.ROTATE_180
    else:
        rotation = cv.ROTATE_90_COUNTERCLOCKWISE
    new_sides = []
    for side in np.roll(sides, count):
        l_corner = new_pixel(side[0][1], side[0][0], count * math.pi / 2, tile.shape[1], tile.shape[0])
        r_corner = new_pixel(side[1][1], side[1][0], count * math.pi / 2, tile.shape[1], tile.shape[0])
        new_sides.append((l_corner, r_corner))
    new_tile = cv.rotate(tile, rotation)
    return new_tile, new_sides

def rotate(tile, corner, count):
    count = count % 4
    if count == 0:
        return tile, (corner[0], corner[1])
    elif count == 1:
        rotation = cv.ROTATE_90_CLOCKWISE
    elif count == 2:
        rotation = cv.ROTATE_180
    else:
        rotation = cv.ROTATE_90_COUNTERCLOCKWISE
    new_corner = new_pixel(corner[1], corner[0], count * math.pi / 2, tile.shape[1], tile.shape[0])
    new_tile = cv.rotate(tile, rotation)
    return new_tile, new_corner

def rotate_point(corner, count, shape=(300, 300)):
    return new_pixel(corner[1], corner[0], (count % 4) * math.pi / 2, shape[1], shape[0])

def is_flat_side(edge_dist_feat, thresh=20):
    return np.absolute(edge_dist_feat - edge_dist_feat[0]).sum() < thresh

def cosine_sim(vec0, vec1):
    return np.dot(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1))

def dot_sim(vec0, vec1):
    return np.dot(vec0, vec1)

def one_to_one_sim(vec0, vec1):
    return np.add(vec0, vec1).sum()

def side_len(corners):
    return math.dist(corners[0], corners[1])

def combine_angle(corners_src, side_src, corners_tgt, side_tgt):
    startpoint_up = rotate_point(corners_src[1], TILE_SRC_ANCHOR - side_src)
    endpoint_up = rotate_point(corners_src[0], TILE_SRC_ANCHOR - side_src)
    startpoint_down = rotate_point(corners_tgt[0], TILE_TGT_ANCHOR - side_tgt)
    endpoint_down = rotate_point(corners_tgt[1], TILE_TGT_ANCHOR - side_tgt)
    endpoint_up = (endpoint_up[0] - startpoint_up[0], endpoint_up[1] -  startpoint_up[1])
    endpoint_down = (endpoint_down[0] - startpoint_down[0], endpoint_down[1] -  startpoint_down[1])
    # start_up = cv.circle(src.copy(), (startpoint_up[1], startpoint_up[0]), 5, (127, 127, 127), 3)
    # end_up = cv.circle(src.copy(), (endpoint_up[1], endpoint_up[0]), 20, (127, 127, 127), 3)
    # start_down = cv.circle(tgt.copy(), (startpoint_down[1], startpoint_down[0]), 5, (127, 127, 127), 3)
    # end_down = cv.circle(tgt.copy(), (endpoint_down[1], endpoint_down[0]), 20, (127, 127, 127), 3)
    # print(fc.angle(endpoint_up, (0, 0), endpoint_down))
    # print(endpoint_up)
    # print(endpoint_down)
    # cv.imshow("sup", start_up)
    # cv.imshow("eup", end_up)
    # cv.imshow("sdwn", start_down)
    # cv.imshow("edwn", end_down)
    # cv.waitKey(0)
    return fc.angle(endpoint_up, (0, 0), endpoint_down)
    
def combine_on_right_or_down(anch, anch_row, anch_col, tgt, tgt_row, tgt_col):
    row_len = max(anch_row + tgt.shape[0] - tgt_row, anch.shape[0])
    col_len = max(anch_col + tgt.shape[1] - tgt_col, anch.shape[1])
    combined_img = np.zeros((row_len, col_len, 3), np.uint8)
    combined_img[0:anch.shape[0], 0:anch.shape[1]] = anch
    cmb_row = anch_row - tgt_row
    cmb_col = anch_col - tgt_col
    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            if sum(combined_img[cmb_row+i, cmb_col+j]) == 0:
                combined_img[cmb_row+i, cmb_col+j] = tgt[i, j]
            elif sum( tgt[i, j]) != 0:
                combined_img[cmb_row+i, cmb_col+j] = (combined_img[cmb_row+i, cmb_col+j] + tgt[i, j]) / 2
    return combined_img

SIM_INCREASING = 1
SIM_DECREASING = 0
def best_match(unmatched_sides, tile_sides, src_features, tgt_features, sim_func=dot_sim, similarity=SIM_INCREASING):
    best_score = -1e10 if similarity == SIM_INCREASING else 1e10
    best_pair = None
    for src_idx, src_side in unmatched_sides:
        for tgt_idx, tgt_side in unmatched_sides:
            if src_idx == tgt_idx:
                continue
            if abs(side_len(tile_sides[src_idx][src_side]) - side_len(tile_sides[tgt_idx][tgt_side])) > 5:
                continue
            if combine_angle(tile_sides[src_idx][src_side], src_side, tile_sides[tgt_idx][tgt_side], tgt_side) > 2:
                continue
            cur_score = sim_func(src_features[src_idx][src_side], tgt_features[tgt_idx][tgt_side])
            if similarity == SIM_INCREASING and cur_score > best_score:
                best_score = cur_score
                best_pair = ((src_idx, src_side), (tgt_idx, tgt_side))
            if similarity == SIM_DECREASING and cur_score < best_score:
                best_score = cur_score
                best_pair = ((src_idx, src_side), (tgt_idx, tgt_side))
    return best_pair

if __name__ == "__main__":
    img = cv.imread("./res/test0_sample0.png")
    tiles, tile_corners, color_tiles = fc.get_corners(img)
    tile_sides = []
    for i in range(len(tiles)):
        tile = cv.cvtColor(tiles[i], cv.COLOR_GRAY2BGR)
        corners = tile_corners[i]
        sides = get_sides(corners)
        tile_sides.append(sides)

    # Extract features of edges
    edge_dist_feats = []
    edge_color_feats = []
    for src_idx, src in enumerate(tiles):
        edge_dists = []
        edge_colors = []
        for side_idx in range(4):
            dist_feat, color_feat = get_edge_features(src, tile_sides[src_idx][side_idx])
            edge_dists.append(dist_feat)
            edge_colors.append(color_feat)
        edge_dist_feats.append(edge_dists)
        edge_color_feats.append(edge_colors)

    # Tag flat sides
    flat_sides = set()
    for src_idx, src in enumerate(tiles):
        for side_idx in range(4):
            if is_flat_side(edge_dist_feats[src_idx][side_idx]):
                flat_sides.add((src_idx, side_idx))

    # Construct embeddings vectors of edges
    src_embeds = []
    tgt_embeds = []
    for src_idx in range(len(tiles)):
        src_side_embeds = []
        tgt_side_embeds = []
        for src_side in range(4):
            src_edge_vec = edge_dist_feats[src_idx][src_side] 
            src_edge_vec = src_edge_vec - src_edge_vec[0]
            tgt_edge_vec = -np.flip(edge_dist_feats[src_idx][src_side])
            tgt_edge_vec = tgt_edge_vec - tgt_edge_vec[0]
            src_color_vec =  edge_color_feats[src_idx][src_side].flatten()
            src_side_embeds.append(np.concatenate((src_edge_vec, src_color_vec)))
            tgt_side_embeds.append(np.concatenate((tgt_edge_vec, src_color_vec)))
        src_embeds.append(src_side_embeds)
        tgt_embeds.append(tgt_side_embeds)

    # Initial data structure for keeping track of unmatched edges
    unmatched_tile_sides = []
    for i in range(len(tiles)):
        for j in range(4):
            if (i, j) not in flat_sides:
                unmatched_tile_sides.append((i, j))

    # Find matching pairs
    matching_pairs = []
    while len(unmatched_tile_sides):
        match = best_match(unmatched_tile_sides, tile_sides, src_embeds, tgt_embeds)
        if not match:
            break
        src_idx, src_side = match[0]
        tgt_idx, tgt_side = match[1]
        src, tgt = color_tiles[src_idx], color_tiles[tgt_idx]
        combination = combine_vertical_color(src, tile_sides[src_idx][src_side], src_side, tgt, tile_sides[tgt_idx][tgt_side], tgt_side, thickness=100)
        unmatched_tile_sides.remove(match[0])
        unmatched_tile_sides.remove(match[1])
        matching_pairs.append(match)

    # Assemble matches together to form a picture
    # I'll speed this up with referencing data structures later
    # Will brute search for now to make sure this approach works first
    
    islands = []

    def locate_next_match():
        for island_idx, island in enumerate(islands):
            for i, tile_info in enumerate(island):
                _, _, tile_idx, _ = tile_info
                for match in matching_pairs:
                    if tile_idx == match[0][0]:
                        return island_idx, i, match
        islands.append([(0, 0, matching_pairs[0][0][0], 0)])
        return len(islands)-1, 0, matching_pairs[0]

    matched_locs = {}
    while len(matching_pairs) > 0:
        island_idx, arr_idx, match = locate_next_match()
        src_idx, src_side = match[0]
        tgt_idx, tgt_side = match[1]
        row, col, _, top = islands[island_idx][arr_idx]
        move_dir = rotate_point((1, 0), top - src_side, shape=(0, 0))
        if tgt_idx not in matched_locs or matched_locs[tgt_idx] == (move_dir[0] + row, move_dir[1] + col):
            islands[island_idx].append((move_dir[0] + row, move_dir[1] + col, tgt_idx, (2 - src_side + tgt_side - top) % 4))
            matched_locs[tgt_idx] = (move_dir[0] + row, move_dir[1] + col)
        matching_pairs.remove(match)
    
    print(islands)

    post_proc_tiles = {}
    for island in islands:
        for tile in island:
            row, col, tile_idx, top_side = tile
            tile_up, rotated_sides = rotate_complete(color_tiles[tile_idx], tile_sides[tile_idx], top_side)
            if tile_idx not in post_proc_tiles:
                post_proc_tiles[tile_idx] = (tile_up, rotated_sides)

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    def src_relevant_corner(sides, dir):
        return sides[dir][0]
    
    def tgt_relevant_corner(sides, dir):
        return sides[dir][1]


    combined_tile_info = {}
    relevant_island = sorted(islands[0], key=lambda x: (-x[0], x[1]))
    anch, anch_sides = post_proc_tiles[relevant_island[0][2]]

    def update_info(row, col, sides):
        combined_tile_info[(row + 1, col)] = (src_relevant_corner(sides, UP), DOWN)
        combined_tile_info[(row - 1, col)] = (src_relevant_corner(sides, DOWN), UP)
        combined_tile_info[(row, col + 1)] = (src_relevant_corner(sides, RIGHT), LEFT)
        combined_tile_info[(row, col - 1)] = (src_relevant_corner(sides, LEFT), RIGHT)

    print(islands[0])
    print(relevant_island)
    combined_img = anch
    update_info(relevant_island[0][0], relevant_island[0][1], anch_sides)
    del relevant_island[0]
    while len(relevant_island) > 0:
        cv.imshow("cmb", combined_img)
        cv.waitKey(0)
        for i, tile in enumerate(relevant_island):
            if (tile[0], tile[1]) in combined_tile_info:
                print(tile)
                corner, dir = combined_tile_info[(tile[0], tile[1])]
                tgt, tgt_sides = post_proc_tiles[tile[2]]
                tgt_corner = tgt_relevant_corner(tgt_sides, dir)
                combined_img = combine_on_right_or_down(combined_img, corner[0], corner[1], tgt, tgt_corner[0], tgt_corner[1])
                row_off = corner[0] - tgt_corner[0]
                col_off = corner[1] - tgt_corner[1]
                new_sides = []
                for side in tgt_sides:
                    new_sides.append((side[0] + row_off, side[1] + col_off))
                update_info(tile[0], tile[1], new_sides)
                del relevant_island[i]
                break


    # for src_idx, src in enumerate(tiles):
    #     for tgt_idx, tgt in enumerate(tiles):
    #         if src_idx == tgt_idx:
    #             continue
    #         for src_side in range(4):
    #             if is_flat_side(edge_dist_feats[src_idx][src_side]):
    #                 continue
    #             src_edge_vec = edge_dist_feats[src_idx][src_side] 
    #             src_edge_vec = src_edge_vec - src_edge_vec[0]
    #             src_color_vec =  edge_color_feats[src_idx][src_side].flatten()
    #             src_vec = np.concatenate((src_edge_vec, src_color_vec))
    #             for tgt_side in range(4):
    #                 if is_flat_side(edge_dist_feats[tgt_idx][tgt_side]):
    #                     continue
    #                 if abs(side_len(tile_sides[src_idx][src_side]) - side_len(tile_sides[tgt_idx][tgt_side])) > 5:
    #                     continue
    #                 if combine_angle(tile_sides[src_idx][src_side], src_side, tile_sides[tgt_idx][tgt_side], tgt_side, src, tgt) > 2:
    #                     continue
    #                 tgt_edge_vec = -np.flip(edge_dist_feats[tgt_idx][tgt_side])
    #                 tgt_edge_vec = tgt_edge_vec - tgt_edge_vec[0]
    #                 tgt_color_vec = edge_color_feats[tgt_idx][tgt_side].flatten()
    #                 tgt_vec = np.concatenate((tgt_edge_vec, tgt_color_vec))
    #                 combinations.append((dot_sim(src_vec, tgt_vec), (src_idx, src_side), (tgt_idx, tgt_side)))

    # combinations.sort(key=lambda x: x[0], reverse=True)
    # for i in range(min(5, len(combinations))):
    #     src_idx, src_side = combinations[i][1]
    #     tgt_idx, tgt_side = combinations[i][2]
    #     src = color_tiles[src_idx]
    #     tgt = color_tiles[tgt_idx]
    #     combination = combine_vertical_color(src, tile_sides[src_idx][src_side], src_side, tgt, tile_sides[tgt_idx][tgt_side], tgt_side, thickness=100)
    #     # src_vec = edge_dist_feats[src_idx][src_side]
    #     # tgt_vec = -np.flip(edge_dist_feats[tgt_idx][tgt_side])
    #     # print(i, "\n--------")
    #     # print(src_vec)
    #     # print(src_vec - src_vec[0])
    #     # print(tgt_vec)
    #     # print(tgt_vec - tgt_vec[0])
    #     cv.imshow(f"Combination{i}", combination)
    # cv.waitKey(0)