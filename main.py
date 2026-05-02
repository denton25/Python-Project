import torch
import numpy as np
import cv2

from cnn_torch import CNN

# ============================================================
# 47‑CLASS EMNIST BALANCED MAPPING
# ============================================================

IDX_TO_CHAR = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]
assert len(IDX_TO_CHAR) == 47


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(device="cpu"):
    device = torch.device(device)

    model = CNN(num_classes=47)
    state = torch.load("model.pth", map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# NORMALIZE CHARACTER
# ============================================================
def normalize_char(char_img):
    # 1. Crop to bounding box (remove extra whitespace)
    ys, xs = np.where(char_img < 200)  # dark pixels
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    char_img = char_img[y1:y2+1, x1:x2+1]

    # 2. Pad to square
    h, w = char_img.shape
    side = max(h, w)
    padded = np.full((side, side), 255, dtype=np.uint8)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = char_img

    # 3. Resize to 28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. Normalize like training
    resized = resized.astype("float32") / 255.0
    resized = (resized - 0.5) / 0.5

    return resized

# ============================================================
# PREPROCESS (OTSU + DILATION)
# ============================================================
def preprocess_document(gray):
    # 1. Deskew using moments
    coords = np.column_stack(np.where(gray < 255))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

    # 2. Otsu threshold
    _, bin_img = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3. Morphological closing to fix broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # 4. Remove small noise
    bin_img = cv2.medianBlur(bin_img, 3)

    return bin_img

# ============================================================
# LINE SEGMENTATION
# ============================================================
def segment_lines(bin_img):
    # Dilate horizontally to merge text into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    dilated = cv2.dilate(bin_img, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:  # ignore tiny noise
            lines.append((y, y + h))

    lines = sorted(lines, key=lambda b: b[0])
    return lines

# ============================================================
# WORD SEGMENTATION
# ============================================================
def segment_words(line_img):
    # Dilate to merge characters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(line_img, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 8 and h > 10:
            boxes.append((x, y, w, h))

    return sorted(boxes, key=lambda b: b[0])

# ============================================================
# CHARACTER SEGMENTATION
# ============================================================
def segment_characters(word_img):
    contours, _ = cv2.findContours(
        word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise
        if w < 3 or h < 8:
            continue

        # Keep accents (very small but above letters)
        if h < 12 and y < 5:
            pass

        boxes.append((x, y, w, h))

    return sorted(boxes, key=lambda b: b[0])


# ============================================================
# FULL SEGMENTATION PIPELINE
# ============================================================
def segment_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return [], []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = preprocess_document(gray)

    lines = segment_lines(bin_img)

    all_chars = []
    all_boxes = []

    for (y1, y2) in lines:
        line_img = bin_img[y1:y2, :]

        words = segment_words(line_img)

        for (wx, wy, ww, wh) in words:
            word_img = line_img[wy:wy+wh, wx:wx+ww]

            chars = segment_characters(word_img)

            for (cx, cy, cw, ch) in chars:
                char_img = word_img[cy:cy+ch, cx:cx+cw]

                # Normalize to 28x28
                side = max(cw, ch)
                padded = np.zeros((side, side), dtype=np.uint8)
                yoff = (side - ch) // 2
                xoff = (side - cw) // 2
                padded[yoff:yoff+ch, xoff:xoff+cw] = char_img

                padded = normalize_char(padded)
                all_chars.append(padded)

                abs_x = wx + cx
                abs_y = y1 + cy
                all_boxes.append((abs_x, abs_y, cw, ch))

    return all_chars, all_boxes

# ============================================================
# PREDICT SINGLE CHARACTER
# ============================================================
def predict_char(model, device, char_img):
    char_img = char_img.astype(np.float32) / 255.0
    tensor = torch.tensor(char_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()

    return pred


# ============================================================
# PREDICT FULL IMAGE
# ============================================================
def predict_word_fn(model, device, image_path):
    chars, boxes = segment_image(image_path)

    if not chars:
        return "", []

    indices = [predict_char(model, device, ch) for ch in chars]
    text = "".join(IDX_TO_CHAR[i] for i in indices)

    return text, boxes











