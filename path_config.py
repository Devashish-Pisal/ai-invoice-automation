from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TRAIN_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "SROIE2019" / "train"
TEST_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "SROIE2019" / "test"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_DATA_PATH = PROJECT_ROOT / "model_data"

