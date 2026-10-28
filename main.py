import cv2 as cv
import numpy as np

#Define constants
w = 100
h = 100

# Define lists
tiles = []
tilesblurred = []
rows = []
hsv_values = []
tile_types = []
rectangles = []

# Image setup
# To change the image, simply enter the corresponding file name inside the parentheses, like this: ("file.name"), on the line below.
img = cv.imread("24.jpg")
display_img = img.copy() #Creates a copy of the original image to draw on

# Crown setup
crown = cv.imread("crown_2.png")
crown_threshold = 0.6

    #resize the crown template to a fixed size
resized_crown = cv.resize(crown, (30, 23))

# Create 4 rotations of the crown template
crown_0 = resized_crown
crown_90 = cv.rotate(resized_crown, cv.ROTATE_90_CLOCKWISE)
crown_180 = cv.rotate(resized_crown, cv.ROTATE_180)
crown_270 = cv.rotate(resized_crown, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Create a list of crown templates for each rotation
crown_templates = [
    crown_0,
    crown_90,
    crown_180,
    crown_270
]

def detect_crowns(img, crown_template):
# This function detects crowns in the given image using template matching with the provided crown template.

    rectangles = []

    # Run template matching for each rotation
    for template in crown_templates:
        hCrown, wCrown = template.shape[:2]

        crown_result = cv.matchTemplate(
            img,
            template,
            cv.TM_CCOEFF_NORMED
        )

        yloc, xloc = np.where(crown_result >= crown_threshold)

        for (x, y) in zip(xloc, yloc):
            rect = [
                int(x),
                int(y),
                int(wCrown),
                int(hCrown)
            ]

            rectangles.append(rect)
            rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(
        rectangles,
        1,
        0.4
    )

    return rectangles

def split_tiles(img):
    # This function splits the given image into tiles of size w x h and returns a list of tiles.
    tiles = []
    for y in range(0, img.shape[0], h):
        for x in range(0, img.shape[1], w):
            tile = img[y:y+h, x:x+w]
            tiles.append(tile)
    return tiles

def process_tiles(tiles):
    # This function processes the given list of tiles and returns a list of HSV values for each tile.
    hsv_values = []
    tilesblurred = []

    for tile in tiles:

        tile = tile.copy()

        top = tile[:20, :]
        bottom = tile[80:, :]
        left = tile[20:80, :20]
        right = tile[20:80, 80:]

        top_pixels = top.reshape(-1, 3)
        bottom_pixels = bottom.reshape(-1, 3)
        left_pixels = left.reshape(-1, 3)
        right_pixels = right.reshape(-1, 3)

        edge_pixels = np.concatenate(
            (
                top_pixels,
                bottom_pixels,
                left_pixels,
                right_pixels
            ),
            axis=0
        )

        mean_color = np.mean(
            edge_pixels,
            axis=0
        ).astype(np.uint8)

        tile[20:80, 20:80] = mean_color

        blurred = cv.GaussianBlur(
            tile,
            (99, 99),
            0
        )

        tilesblurred.append(blurred)

        hsv_tile = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        h_val = np.mean(hsv_tile[:, :, 0])
        s_val = np.mean(hsv_tile[:, :, 1])
        v_val = np.mean(hsv_tile[:, :, 2])

        hsv_values.append((h_val, s_val, v_val))

    return tilesblurred, hsv_values

def classify_tiles(hsv_values):
    # This function classifies the given list of HSV values into tile types based on predefined thresholds.
    tile_types = []
    for i, (h_val, s_val, v_val) in enumerate(hsv_values):
        if 95 <= h_val <= 115:
            group = "Ocean"
        elif 22 <= h_val <= 30 and s_val >= 190 and v_val >= 130:
            group = "Farm"
        elif 34 <= h_val <= 50 and s_val >= 160 and v_val >= 105:
            group = "Plains"
        elif 30 <= h_val <= 55 and s_val >= 120 and v_val < 115:
            group = "Forrest"
        elif 15 <= h_val <= 28 and v_val < 70:
            group = "Mine"
        elif 15 <= h_val <= 30 and 70 <= v_val <= 140:
            group = "Wasteland"
        else:
            group = "Unknown"

        tile_types.append(group)
        print(i, group, h_val, s_val, v_val)

    return tile_types

def find_groups(tile_types):
    # This function finds the groups of tiles based on their types and returns a list of groups.

    grid = np.array(tile_types).reshape(5, 5)

    visited = np.zeros(
        (5, 5),
        dtype=bool
    )

    group_ids = np.full(
        (5, 5),
        -1,
        dtype=int
    )

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    current_group = 0

    for row in range(5):
        for col in range(5):

            if visited[row, col]:
                continue

            tile_type = grid[row, col]

            stack = [(row, col)]

            visited[row, col] = True
            group_ids[row, col] = current_group

            while stack:

                r, c = stack.pop()

                for dr, dc in directions:

                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < 5 and 0 <= nc < 5:

                        if (
                            not visited[nr, nc]
                            and grid[nr, nc] == tile_type
                        ):

                            visited[nr, nc] = True
                            group_ids[nr, nc] = current_group

                            stack.append((nr, nc))

            current_group += 1

    return group_ids

def calculate_score(groups,rectangles):
    # This function calculates the score based on the groups of tiles and their positions.

    tiles_per_group = {}
    crowns_per_group = {}
    scores = {}

    for row in range(5):
        for col in range(5):

            group_id = groups[row, col]

            if group_id not in tiles_per_group:
                tiles_per_group[group_id] = 0

            tiles_per_group[group_id] += 1

    for (x, y, wCrown, hCrown) in rectangles:

        center_x = x + wCrown // 2
        center_y = y + hCrown // 2

        col = center_x // 100
        row = center_y // 100

        if 0 <= row < 5 and 0 <= col < 5:

            group_id = groups[row, col]

            if group_id not in crowns_per_group:
                crowns_per_group[group_id] = 0

            crowns_per_group[group_id] += 1

    for group_id in tiles_per_group:

        tile_count = tiles_per_group[group_id]
        crown_count = crowns_per_group.get(group_id, 0)

        score = tile_count * crown_count

        scores[group_id] = score

        print(
            "Group:",
            group_id,
            "Tiles:",
            tile_count,
            "Crowns:",
            crown_count,
            "Score:",
            score
        )

    total_score = sum(scores.values())

    return scores, total_score

def display_results(img, tilesblurred, tile_types, group_ids, rectangles):

    display_img = img.copy()

    # Draw crown detections
    for (x, y, wRect, hRect) in rectangles:
        cv.rectangle(
            display_img,
            (x, y),
            (x + wRect, y + hRect),
            (0, 255, 0),
            1
        )

    # Write tile type and group id
    for i, tile_type in enumerate(tile_types):

        row = i // 5
        col = i % 5

        x = col * 100
        y = row * 100

        cv.putText(
            display_img,
            tile_type,
            (x + 5, y + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv.LINE_AA
        )

        group_id = group_ids[row, col]

        cv.putText(
            display_img,
            f"G{group_id}",
            (x + 5, y + 45),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        cv.LINE_AA
        
        )

    # Reconstruct blurred tiles
    rows = []

    for start in range(0, 25, 5):

        row = np.concatenate(
            tilesblurred[start:start+5],
            axis=1
        )

        rows.append(row)

    reconstructed = np.concatenate(
        rows,
        axis=0
    )

    # Blur full original image for comparison
    blurredorg = cv.GaussianBlur(
        img,
        (99, 99),
        0
    )

    # Show images
    cv.imshow(
        "Classification + Groups + Crowns",
        display_img
    )

    cv.imshow(
        "Reconstructed",
        reconstructed
    )

    cv.imshow(
        "Original",
        img
    )

    cv.imshow(
        "Blurred Original",
        blurredorg
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


# Main execution

rectangles = detect_crowns(img, "crown_2.png")

tiles = split_tiles(img)

tilesblurred, hsv_values = process_tiles(tiles)

tile_types = classify_tiles(hsv_values)

group_ids = find_groups(tile_types)

scores, total_score = calculate_score(
    group_ids,
    rectangles
)

print("Total score:", total_score)

display_results(
    img,
    tilesblurred,
    tile_types,
    group_ids,
    rectangles
)
