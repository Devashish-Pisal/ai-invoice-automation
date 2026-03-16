from path_config import TRAIN_DATASET_PATH, TEST_DATASET_PATH
from PIL import Image
from loguru import logger
import json


def parse_sroie_sample(img_path, box_path, entity_path):
    """
    Create words, bounding boxes and labels from the image, box data, and entity data.
    """
    with Image.open(img_path) as img:
        width, height = img.size

    words = []
    boxes = []
    # Parse OCR
    with open(box_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",", 8)
            quad = list(map(int, parts[:8]))
            word = ",".join(parts[8:]).strip()
            rect = quad_to_bbox(quad)
            # normalize
            norm_box = [
                int(1000 * rect[0] / width),
                int(1000 * rect[1] / height),
                int(1000 * rect[2] / width),
                int(1000 * rect[3] / height),
            ]
            words.append(word)
            boxes.append(norm_box)

    # Parse entities
    with open(entity_path, "r", encoding="utf-8") as file:
        gt_dict = json.load(file)
    # Assign labels
    labels = ["O"] * len(words)
    for field, value in gt_dict.items():
        entity_tokens = value.split()
        for i, token in enumerate(entity_tokens):
            for j, w in enumerate(words):
                if w == token:
                    labels[j] = f"B-{field.upper()}" if i == 0 else f"I-{field.upper()}"
    if len(words) == 0 or len(boxes) == 0 or len(labels) == 0:
        logger.error(f"Empty words/boxes/labels: {img_path}, {box_path}, {entity_path}")

    return words, boxes, labels



def quad_to_bbox(quad):
    """
    Convert quadrilateral [x0,y0,x1,y1,x2,y2,x3,y3] to rectangle [x_min, y_min, x_max, y_max]
    """
    xs = quad[0::2]
    ys = quad[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]
