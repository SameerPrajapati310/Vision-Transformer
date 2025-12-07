import torch
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from PIL import Image
from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Load the VQA v2 dataset (subset for quick testing)
dataset = load_dataset("hf-internal-testing/fixtures_vqa")

# Load processor and model
model_id = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)

# LoRA config for PEFT
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # ViLT uses BertSelfAttention internally
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS  # Closest match for VQA
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Preprocessing function
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    encoding = processor(image, example["question"], padding="max_length", truncation=True, return_tensors="pt")
    encoding["labels"] = torch.tensor(example["answer_label"])
    return {k: v.squeeze() for k, v in encoding.items()}

# Apply preprocessing
processed_dataset = dataset["train"].map(preprocess)

# DataLoader
def collate_fn(batch):
    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
        "attention_mask": torch.stack([example["attention_mask"] for example in batch]),
        "labels": torch.tensor([example["labels"] for example in batch])
    }

train_loader = DataLoader(processed_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./vilt-lora-vqa",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    learning_rate=2e-4,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor,
    data_collator=collate_fn
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./vilt-lora-vqa-lora")
