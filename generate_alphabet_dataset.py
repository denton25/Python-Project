import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

OUTPUT_DIR = "printed_letters_dataset"
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

FONT_DIR = "fonts"

FONTS = [
    os.path.join(FONT_DIR, f)
    for f in os.listdir(FONT_DIR)
    if f.lower().endswith((".ttf", ".otf"))
]

IMAGES_PER_CLASS = 1000  # bump up if you want more


def generate_char_image(char, font_path):
    # random font size
    font_size = random.randint(24, 56)
    font = ImageFont.truetype(font_path, font_size)

    # random light background (to mimic highlights)
    bg = random.randint(220, 255)
    img = Image.new("L", (96, 96), color=bg)
    draw = ImageDraw.Draw(img)

    # random position
    x = random.randint(5, 25)
    y = random.randint(5, 25)

    # draw character (black)
    draw.text((x, y), char, font=font, fill=0)

    # convert to numpy
    img = np.array(img)

    # random small rotation
    angle = random.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((48, 48), angle, 1.0)
    img = cv2.warpAffine(img, M, (96, 96), borderValue=bg)

    # random noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # crop to bounding box
    ys, xs = np.where(img < 200)
    if len(xs) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        img = img[y1:y2+1, x1:x2+1]

    # pad to square
    h, w = img.shape
    side = max(h, w)
    padded = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = img

    # resize to 28x28
    img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    return img


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
    print("Alphabet dataset generation complete!")
