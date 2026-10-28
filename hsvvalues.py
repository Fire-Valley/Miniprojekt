import cv2 as cv
import numpy as np

w = 100
h = 100

tiles = []
tilesblurred = []
rows = []
hsv_values = []

img = cv.imread("5.jpg")

# 1. Skær billedet op i 25 tiles
for y in range(0, 500, 100):
    for x in range(0, 500, 100):
        tile = img[y:y+h, x:x+w].copy()
        tiles.append(tile)


# 2. Behandl hvert tile
for tile in tiles:

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

    # Konverter til normal pixel-type
    mean_color = mean_color.astype(np.uint8)

    # Erstat midten 60x60
    tile[20:80, 20:80] = mean_color

    # Blur tile'et
    blurred = cv.GaussianBlur(tile, (99, 99), 0)

    # Gem det færdige tile
    tilesblurred.append(blurred)

    # -------------------------------
    # HSV FRA DET SLØREDE TILE
    # -------------------------------

    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    mean_hsv = cv.mean(hsv)

    h_mean, s_mean, v_mean = mean_hsv[:3]

    hsv_values.append((h_mean, s_mean, v_mean))


# 3. Print HSV-værdier
for i, (h, s, v) in enumerate(hsv_values):
    print(i, h, s, v)


# 4. Saml 5 tiles ad gangen til rækker
for start in range(0, 25, 5):
    row = np.concatenate(
        tilesblurred[start:start+5],
        axis=1
    )
    rows.append(row)


# 5. Saml rækkerne til hele billedet
reconstructed = np.concatenate(rows, axis=0)


# 6. Vis resultat
cv.imshow("Reconstructed", reconstructed)
cv.imshow("Original", img)

cv.waitKey(0)
cv.destroyAllWindows()