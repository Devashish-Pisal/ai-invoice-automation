from huggingface_hub import login, upload_folder
from dotenv import load_dotenv
from loguru import logger
from path_config import MODEL_DATA_PATH
import os


load_dotenv()
HUGGINGFACE_TOKEN = os.environ.get("HF_WRITE_TOKEN")
if HUGGINGFACE_TOKEN:
    login(HUGGINGFACE_TOKEN)
    logger.info("Hugging Face authentication successful.")


FINAL_MODEL_PATH = os.path.join(MODEL_DATA_PATH, "layoutlmv3-sroie-final")
upload_folder(
    repo_id="devashish-pisal/layoutlmv3-sroie-token-classification",
    folder_path=FINAL_MODEL_PATH,
    commit_message="upload model",
    repo_type="model"
)
logger.success("Model uploaded!")