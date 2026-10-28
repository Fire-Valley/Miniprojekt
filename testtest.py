import cv2 as cv
import numpy as np

w = 100
h = 100

tiles = []
tilesblurred = []
rows = []
hsv_values = []
tile_types = []

img = cv.imread("67.jpg")
display_img = img.copy()

# -----------------------------
# CROWN TEMPLATE MATCHING
# -----------------------------

crown = cv.imread("crown_2.png")
resized_crown = cv.resize(crown, (30, 23))

# Lav 4 rotationer
crown_0 = resized_crown
crown_90 = cv.rotate(resized_crown, cv.ROTATE_90_CLOCKWISE)
crown_180 = cv.rotate(resized_crown, cv.ROTATE_180)
crown_270 = cv.rotate(resized_crown, cv.ROTATE_90_COUNTERCLOCKWISE)

crown_templates = [
    crown_0,
    crown_90,
    crown_180,
    crown_270
]

crown_threshold = 0.6
rectangles = []

# Kør template matching for hver rotation
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

        # Dublet så groupRectangles virker
        rectangles.append(rect)
        rectangles.append(rect)

print("Matches before grouping:", len(rectangles))

rectangles, weights = cv.groupRectangles(
    rectangles,
    1,
    0.4
)

print("Crowns after grouping:", len(rectangles))

# Tegn crowns
for (x, y, wRect, hRect) in rectangles:

    cv.rectangle(
        display_img,
        (x, y),
        (x + wRect, y + hRect),
        (0, 255, 0),
        1
    )


# -----------------------------
# 1. SKÆR BILLEDET OP I 25 TILES
# -----------------------------

for y in range(0, 500, 100):
    for x in range(0, 500, 100):

        tile = img[y:y+h, x:x+w].copy()
        tiles.append(tile)


# -----------------------------
# 2. BEHANDL HVERT TILE
# -----------------------------

for tile in tiles:

    top = tile[:20, :]
    bund = tile[80:, :]
    venstre = tile[20:80, :20]
    hoejre = tile[20:80, 80:]

    top_pixels = top.reshape(-1, 3)
    bund_pixels = bund.reshape(-1, 3)
    venstre_pixels = venstre.reshape(-1, 3)
    hoejre_pixels = hoejre.reshape(-1, 3)

    kant_pixels = np.concatenate(
        (
            top_pixels,
            bund_pixels,
            venstre_pixels,
            hoejre_pixels
        ),
        axis=0
    )

    mean_color = np.mean(
        kant_pixels,
        axis=0
    ).astype(np.uint8)

    # Erstat midten
    tile[20:80, 20:80] = mean_color

    # Blur
    blurred = cv.GaussianBlur(
        tile,
        (99, 99),
        0
    )

    tilesblurred.append(blurred)

    # HSV
    hsv = cv.cvtColor(
        blurred,
        cv.COLOR_BGR2HSV
    )

    mean_hsv = cv.mean(hsv)

    h_mean, s_mean, v_mean = mean_hsv[:3]

    hsv_values.append(
        (h_mean, s_mean, v_mean)
    )


# -----------------------------
# 3. KLASSIFICER TILES
# -----------------------------

for i, (h_val, s_val, v_val) in enumerate(hsv_values):

    if 95 <= h_val <= 115:
        gruppe = "Ocean"

    elif 22 <= h_val <= 30 and s_val >= 190 and v_val >= 130:
        gruppe = "Farm"

    elif 34 <= h_val <= 50 and s_val >= 160 and v_val >= 105:
        gruppe = "Plains"

    elif 30 <= h_val <= 55 and s_val >= 120 and v_val < 115:
        gruppe = "Forrest"

    elif 15 <= h_val <= 28 and v_val < 70:
        gruppe = "Mine"

    elif 15 <= h_val <= 30 and 70 <= v_val <= 140:
        gruppe = "Wasteland"

    else:
        gruppe = "Unknown"

    tile_types.append(gruppe)

    row = i // 5
    col = i % 5

    x = col * 100
    y = row * 100

    cv.putText(
        display_img,
        gruppe,
        (x + 5, y + 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv.LINE_AA
    )

    print(i, gruppe, h_val, s_val, v_val)


# -----------------------------
# 4. LAV 5x5 GRID
# -----------------------------

grid = np.array(
    tile_types
).reshape(5, 5)

print("\nTile types:")
print(grid)


# -----------------------------
# 5. FIND SAMMENHÆNGENDE GRUPPER
# -----------------------------

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


print("\nGroup IDs:")
print(group_ids)


# -----------------------------
# 6. SKRIV GROUP_ID PÅ BILLEDET
# -----------------------------

for row in range(5):
    for col in range(5):

        group_id = group_ids[row, col]

        x = col * 100
        y = row * 100

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


# -----------------------------
# 7. TÆL TILES PER GROUP
# -----------------------------

tiles_per_group = {}

for row in range(5):
    for col in range(5):

        group_id = group_ids[row, col]

        if group_id not in tiles_per_group:
            tiles_per_group[group_id] = 0

        tiles_per_group[group_id] += 1


# -----------------------------
# 8. TÆL CROWNS PER GROUP
# -----------------------------

crowns_per_group = {}

for (x, y, wRect, hRect) in rectangles:

    # Find centrum af crown
    center_x = x + wRect // 2
    center_y = y + hRect // 2

    # Find tile-position
    col = center_x // 100
    row = center_y // 100

    # Sikkerhed så vi ikke går udenfor grid
    if 0 <= row < 5 and 0 <= col < 5:

        group_id = group_ids[row, col]

        if group_id not in crowns_per_group:
            crowns_per_group[group_id] = 0

        crowns_per_group[group_id] += 1


print("\nCrowns per group:")
print(crowns_per_group)


# -----------------------------
# 9. BEREGN SCORE
# -----------------------------

scores = {}

print("\nScores:")

for group_id in tiles_per_group:

    antal_tiles = tiles_per_group[group_id]

    antal_crowns = crowns_per_group.get(
        group_id,
        0
    )

    score = antal_tiles * antal_crowns

    scores[group_id] = score

    print(
        "Group", group_id,
        "| Type:", grid[
            np.argwhere(group_ids == group_id)[0][0],
            np.argwhere(group_ids == group_id)[0][1]
        ],
        "| Tiles:", antal_tiles,
        "| Crowns:", antal_crowns,
        "| Score:", score
    )


# Samlet score
total_score = sum(scores.values())

print("\nTOTAL SCORE:", total_score)


# -----------------------------
# 10. SAML TILES TIL RÆKKER
# -----------------------------

for start in range(0, 25, 5):

    row = np.concatenate(
        tilesblurred[start:start+5],
        axis=1
    )

    rows.append(row)


# -----------------------------
# 11. SAML HELE BILLEDET
# -----------------------------

reconstructed = np.concatenate(
    rows,
    axis=0
)


# -----------------------------
# 12. BLUR ORIGINAL
# -----------------------------

blurredorg = cv.GaussianBlur(
    img,
    (99, 99),
    0
)


# -----------------------------
# 13. VIS BILLEDER
# -----------------------------

cv.imshow(
    "Klassifikation + grupper + crowns",
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

cv.imshow(
    "Crown",
    resized_crown
)

cv.waitKey(0)
cv.destroyAllWindows()