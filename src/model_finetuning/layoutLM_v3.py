from train_dataset_prep import TRAIN_DATA
from test_dataset_prep import TEST_DATA
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from dataset_util import preprocess
from datasets import Dataset
from path_config import PROCESSED_DATA_PATH, MODEL_DATA_PATH
from loguru import logger
from huggingface_hub import login
from dotenv import load_dotenv
import numpy as np
import os
import torch
import evaluate
import time

# Login to HF
load_dotenv()
HUGGINGFACE_TOKEN = os.environ.get("HF_WRITE_TOKEN")
if HUGGINGFACE_TOKEN:
    login(HUGGINGFACE_TOKEN)
    logger.info("Hugging Face authentication successful.")

# Collect unique labels from the datasets, so that we can assign them numbers (ids)
unique_labels = set()
for sample in TRAIN_DATA:
    unique_labels.update(sample['labels'])
for sample in TEST_DATA:
    unique_labels.update(sample['labels'])
unique_labels = sorted(list(unique_labels))

# assign IDs to the labels & create mappings for model input
label2id = {k:v for v, k in enumerate(unique_labels)}
id2label = {k:v for v, k in label2id.items()}

# Load processor (It combines image processor and tokenizer).
processor = LayoutLMv3Processor.from_pretrained( "microsoft/layoutlmv3-base", apply_ocr=False)
# Load base model
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

ENCODED_TRAIN_DATASET = None
ENCODED_TEST_DATASET = None
encoded_train_dataset_path = PROCESSED_DATA_PATH / 'ENCODED_TRAIN_DATASET.pt'
encoded_test_dataset_path = PROCESSED_DATA_PATH / 'ENCODED_TEST_DATASET.pt'
if os.path.exists(encoded_train_dataset_path) and  os.path.exists(encoded_test_dataset_path):
    ENCODED_TRAIN_DATASET = torch.load(encoded_train_dataset_path)
    logger.info("ENCODED_TRAIN_DATASET.pt loaded!")
    ENCODED_TEST_DATASET = torch.load(encoded_test_dataset_path)
    logger.info("ENCODED_TRAIN_DATASET.pt loaded!")
else:
    # Preprocess training dataset
    ENCODED_TRAIN_DATASET = {'input_ids': [], 'attention_mask': [], 'bbox': [], 'labels': [], 'pixel_values': []}
    for sample in TRAIN_DATA:
        encodings = preprocess(sample, label2id, processor)
        ENCODED_TRAIN_DATASET['input_ids'].append(encodings['input_ids'])
        ENCODED_TRAIN_DATASET['attention_mask'].append(encodings['attention_mask'])
        ENCODED_TRAIN_DATASET['bbox'].append(encodings['bbox'])
        ENCODED_TRAIN_DATASET['labels'].append(encodings['labels'])
        ENCODED_TRAIN_DATASET['pixel_values'].append(encodings['pixel_values'])
    torch.save(ENCODED_TRAIN_DATASET, encoded_train_dataset_path)
    logger.success(f"ENCODED_TRAIN_DATASET.pt successfully created!")

    # Preprocess testing dataset
    ENCODED_TEST_DATASET = {'input_ids': [], 'attention_mask': [], 'bbox': [], 'labels': [], 'pixel_values': []}
    for sample in TEST_DATA:
        encodings = preprocess(sample, label2id, processor)
        ENCODED_TEST_DATASET['input_ids'].append(encodings['input_ids'])
        ENCODED_TEST_DATASET['attention_mask'].append(encodings['attention_mask'])
        ENCODED_TEST_DATASET['bbox'].append(encodings['bbox'])
        ENCODED_TEST_DATASET['labels'].append(encodings['labels'])
        ENCODED_TEST_DATASET['pixel_values'].append(encodings['pixel_values'])
    torch.save(ENCODED_TEST_DATASET, encoded_test_dataset_path)
    logger.success(f"ENCODED_TEST_DATASET.pt successfully created!")
ENCODED_TRAIN_DATASET = Dataset.from_dict(ENCODED_TRAIN_DATASET)
ENCODED_TEST_DATASET = Dataset.from_dict(ENCODED_TEST_DATASET)

# Training & Testing Dataset
'''
Dataset({
    features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
    num_rows: 626
})
Dataset({
    features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
    num_rows: 345
})
'''


def compute_metrics(p):
    """
    Use seqeval to compute the token level F1 score.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        curr_preds = []
        curr_labels = []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id == -100:
                # ignore special tokens and padding
                continue
            curr_labels.append(id2label[l_id])
            curr_preds.append(id2label[p_id])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }


#Finetune model
seqeval = evaluate.load("seqeval") # Metric
training_args = TrainingArguments(
    output_dir=MODEL_DATA_PATH,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ENCODED_TRAIN_DATASET,
    eval_dataset=ENCODED_TEST_DATASET,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

logger.info(f"Starting finetuning.....")
start_time = time.time()
trainer.train()
end_time = time.time()
logger.info(f"Finetuning completed!")
logger.info(f"Finetuning took {end_time - start_time} seconds time.")

trainer.save_model(os.path.join(MODEL_DATA_PATH, "layoutlmv3-sroie-final"))
processor.save_pretrained(os.path.join(MODEL_DATA_PATH, "layoutlmv3-sroie-final"))