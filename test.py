import cv2 as cv
import numpy as np

w = 100
h = 100

tiles = []
tilesblurred = []
rows = []
hsv_values = []

img = cv.imread("66.jpg")
display_img = img.copy()

# 1. Skær billedet op i 25 tiles
for y in range(0, 500, 100):
    for x in range(0, 500, 100):
        tile = img[y:y+h, x:x+w].copy()
        tiles.append(tile)

# 2. Behandl hvert tile
for tile in tiles:
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    # 60x60 center betyder 20 pixels kant rundt om
    top = tile[:20, :]
    bund = tile[80:, :]
    venstre = tile[20:80, :20]
    hoejre = tile[20:80, 80:]

    # Lav kantområderne om til pixel-lister
    top_pixels = top.reshape(-1, 3)
    bund_pixels = bund.reshape(-1, 3)
    venstre_pixels = venstre.reshape(-1, 3)
    hoejre_pixels = hoejre.reshape(-1, 3)

    # Saml alle kant-pixels
    kant_pixels = np.concatenate(
        (
            top_pixels,
            bund_pixels,
            venstre_pixels,
            hoejre_pixels
        ),
        axis=0
    )

    # Find gennemsnitsfarven fra kanten
    mean_color = np.mean(kant_pixels, axis=0)

    # Find gennemsnitsfarven i HSV
    mean_hsv = cv.mean(hsv)

    h,s,v = mean_hsv[:3]

    hsv_values.append((h,s,v))



for i, (h, s, v) in enumerate(hsv_values):

    if v < 80:
        gruppe = "mork"

    elif h > 30 and s > 170:
        gruppe = "gron"

    elif s > 180 and v > 145:
        gruppe = "gul"

    elif 21 <= h <= 25 and 115 <= s <= 135 and v > 118:
        gruppe = "special"

    else:
        gruppe = "brun"

    # Find tile-positionen ud fra index
    row = i // 5
    col = i % 5

    x = col * 100
    y = row * 100

    # Skriv klassifikationen oven på tile'et
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


    # Konverter til uint8
    mean_color = mean_color.astype(np.uint8)

    # Erstat midten 60x60 med gennemsnitsfarven
    tile[20:80, 20:80] = mean_color

    # Blur tile'et bagefter
    blurred = cv.GaussianBlur(tile, (99,99), 0)

    tilesblurred.append(blurred)

# 3. Saml 5 tiles ad gangen til rækker
for start in range(0, 25, 5):
    row = np.concatenate(
        tilesblurred[start:start+5],
        axis=1
    )
    rows.append(row)

# 4. Saml rækkerne til hele billedet
reconstructed = np.concatenate(rows, axis=0)

# 5. Blur hele originalbilledet til sammenligning
blurredorg = cv.GaussianBlur(img, (99, 99), 2)

# 6. Vis billederne
cv.imshow("Klassifikation", display_img)
cv.imshow("Reconstructed", reconstructed)
cv.imshow("Original", img)
cv.imshow("Blurred Original", blurredorg)

cv.waitKey(0)
cv.destroyAllWindows()