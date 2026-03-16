from dataset_util import parse_sroie_sample
from path_config import TEST_DATASET_PATH
from datasets import Dataset
from loguru import logger
import os

#Testing folder paths
test_box_folder = TEST_DATASET_PATH / "box"
test_entities_folder = TEST_DATASET_PATH / "entities"
test_img_folder = TEST_DATASET_PATH / "img"


data = {"img_path": [], "words":[], "bboxes":[], "labels":[]}
for img_file_name in os.listdir(test_img_folder):
    base_file_name = img_file_name.rsplit(".", 1)[0]
    box_and_entity_file_name = base_file_name + ".txt"
    img_path = test_img_folder / img_file_name
    box_path = test_box_folder / box_and_entity_file_name
    entities_path = test_entities_folder / box_and_entity_file_name
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


TEST_DATA = Dataset.from_dict(data)
logger.success(f"TEST DATASET is loaded! Length: {len(TEST_DATA)}")
