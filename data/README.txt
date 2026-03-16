- In order to fine tune the layoutLM model on SROIE dataset, the dataset must be downloaded in folder "data/raw/" from kaggle.
- SROIE Dataset download link : https://www.kaggle.com/datasets/urbikn/sroie-datasetv2?select=SROIE2019
- After extracting the downloaded file, the final folder structure of the raw dataset should match following:
- [Training Folder]: data/raw/SROIE2019/train/
- [Testing Folder]: data/raw/SROIE2019/test/

- After running the train_dataset_prep.py and test_dataset_prep.py scripts from folder src/model_finetunning/ json dataset files will be created in data/processed/ folder.
- In this files the image paths are hardcoded (not original images stored), so raw dataset is still required, because while finetuning images are loaded on the fly.