# scripts/04_finetune.py

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os

# 强制使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ===== 配置模型和 tokenizer =====
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# ===== 加载清洗好的 SQuAD 格式数据 =====
dataset = load_dataset("json", data_files={"train": "data/squad/clean_train.json"}, field="data")

# ===== 扁平化成单个样本格式 =====
samples = []
for item in dataset["train"]:
    for paragraph in item["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]
            answer_text = answer["text"]
            answer_start = answer["answer_start"]
            if context[answer_start:answer_start + len(answer_text)] != answer_text:
                continue  # 答案位置不匹配，跳过
            samples.append({
                "context": context,
                "question": question,
                "answer_text": answer_text,
                "answer_start": answer_start
            })

dataset = Dataset.from_list(samples)

# ===== 划分训练集和验证集 =====
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ===== 预处理函数：构建 token 级别的 answer span =====
def preprocess(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second",  # 保证截断发生在 context 部分
        padding="max_length",
        return_offsets_mapping=True
    )

    start_char = example["answer_start"]
    end_char = start_char + len(example["answer_text"])
    offsets = tokenized["offset_mapping"]

    start_token = end_token = None

    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = idx
        if start < end_char <= end:
            end_token = idx

    # fallback：找不到就设为0，防止训练crash
    if start_token is None or end_token is None:
        start_token = end_token = 0

    tokenized.update({
        "start_positions": start_token,
        "end_positions": end_token
    })
    tokenized.pop("offset_mapping")  # 不保留 offset

    return tokenized

# ===== Tokenize 数据 =====
train_tokenized = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
eval_tokenized = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

# ===== 设置训练参数 =====
training_args = TrainingArguments(
    output_dir="models/distilbert-qa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer
)

# ===== 开始训练 =====
print("🚀 开始 fine-tune...")
trainer.train()
tokenizer.save_pretrained("models/distilbert-qa")
print("\n✅ 模型已保存到 models/distilbert-qa")