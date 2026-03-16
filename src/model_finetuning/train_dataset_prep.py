from dataset_util import parse_sroie_sample
from path_config import TRAIN_DATASET_PATH, PROCESSED_DATA_PATH
from datasets import Dataset
from loguru import logger
import os
import json

#Training folder paths
train_box_folder = TRAIN_DATASET_PATH / "box"
train_entities_folder = TRAIN_DATASET_PATH / "entities"
train_img_folder = TRAIN_DATASET_PATH / "img"

TRAIN_DATA = None

if os.path.exists(os.path.join(PROCESSED_DATA_PATH, "TRAIN_DATA.json")):
    with open(os.path.join(PROCESSED_DATA_PATH, "TRAIN_DATA.json")) as f:
        TRAIN_DATA = Dataset.from_dict(json.load(f))
        logger.success("TRAIN_DATA.json loaded!")
else:
    data = {"img_path": [], "words":[], "bboxes":[], "labels":[]}
    for img_file_name in os.listdir(train_img_folder):
        base_file_name = img_file_name.rsplit(".", 1)[0]
        box_and_entity_file_name = base_file_name + ".txt"
        img_path = train_img_folder / img_file_name
        box_path = train_box_folder / box_and_entity_file_name
        entities_path = train_entities_folder / box_and_entity_file_name
        try:
            words, boxes, labels = parse_sroie_sample(img_path, box_path, entities_path)
            if len(words) == len(boxes) == len(labels):
                data["img_path"].append(str(img_path))
                data["words"].append(words)
                data["bboxes"].append(boxes)
                data["labels"].append(labels)
            else:
                logger.warning(f"words/boxes/labels length mismatch, skipping sample: {img_file_name}!")
        except Exception as e:
            logger.warning(f"Skipping {img_file_name} due to error: {e}!")

    with open(os.path.join(PROCESSED_DATA_PATH, "TRAIN_DATA.json"), "w") as f:
        json.dump(data, f, indent=2)
        logger.success("TRAIN_DATA.json successfully created!")
    TRAIN_DATA = Dataset.from_dict(data)

