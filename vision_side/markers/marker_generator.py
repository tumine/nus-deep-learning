"""
Generate all classroom request cards.

Author: DL-V2
"""

import os
import cv2

# ===================================================
# Configuration
# ===================================================

OUTPUT_FOLDER = "markers"

MARKERS = {

    0:"blocks",

    1:"pencil",

    2:"eraser",

    3:"teacher"

}

DICTIONARY = cv2.aruco.DICT_4X4_50

# ===================================================

if not os.path.exists(OUTPUT_FOLDER):

    os.makedirs(OUTPUT_FOLDER)

dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)

print("Generating markers...")

for marker_id,name in MARKERS.items():

    marker = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        300
    )

    marker = cv2.copyMakeBorder(
        marker,
        40,
        40,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255
    )
    filename = f"{OUTPUT_FOLDER}/{name}.png"

    cv2.imwrite(filename,marker)

    print(f"Saved: {filename}")

print("Finished.")