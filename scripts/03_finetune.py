# scripts/03_finetune.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

dataset = load_dataset("json", data_files={"train": "data/squad/train.json"}, field="data")

def preprocess(example):
    return tokenizer(
        example["question"], 
        example["context"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

tokenized = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="models/bible-qa-model",
    evaluation_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
)

trainer.train()
trainer.save_model("models/bible-qa-model")
print("Fine-tuning complete and model saved.")
