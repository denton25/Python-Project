import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "printed_dataset"
CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

FONT_DIR = "fonts"

FONTS = [
    os.path.join(FONT_DIR, f)
    for f in os.listdir(FONT_DIR)
    if f.lower().endswith((".ttf", ".otf"))
]


IMAGES_PER_CLASS = 500  # increase for more accuracy


# ============================================================
# RANDOM IMAGE GENERATION
# ============================================================

def generate_char_image(char, font_path):
    # random font size
    font_size = random.randint(18, 32)
    font = ImageFont.truetype(font_path, font_size)

    # create blank canvas
    img = Image.new("L", (64, 64), color=255)
    draw = ImageDraw.Draw(img)

    # random position
    x = random.randint(5, 20)
    y = random.randint(5, 20)

    # draw character
    draw.text((x, y), char, font=font, fill=0)

    # convert to numpy
    img = np.array(img)

    # random rotation
    angle = random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
    img = cv2.warpAffine(img, M, (64, 64), borderValue=255)

    # random noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # crop to bounding box
    ys, xs = np.where(img < 200)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        img = img[y1:y2+1, x1:x2+1]

    # resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    return img


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for char in CHARS:
        class_dir = os.path.join(OUTPUT_DIR, char)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating: {char}")

        for i in range(IMAGES_PER_CLASS):
            font_path = random.choice(FONTS)
            img = generate_char_image(char, font_path)

            filename = os.path.join(class_dir, f"{char}_{i}.png")
            cv2.imwrite(filename, img)


if __name__ == "__main__":
    generate_dataset()
    print("Dataset generation complete!")
