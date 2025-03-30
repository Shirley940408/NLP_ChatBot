# scripts/03_clean_data.py
import json
from transformers import AutoTokenizer

MAX_ANSWER_TOKENS = 30
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

with open("data/squad/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned_data = {"data": []}
kept_count = 0
skipped_count = 0

for item in data["data"]:
    title = item["title"]
    cleaned_paragraphs = []

    for para in item["paragraphs"]:
        context = para["context"]
        qas = []

        for qa in para["qas"]:
            answer_text = qa["answers"][0]["text"]
            answer_start = qa["answers"][0]["answer_start"]

            # ✅ 1. 答案必须存在于 context 中
            if context[answer_start:answer_start + len(answer_text)] != answer_text:
                skipped_count += 1
                continue

            # ✅ 2. 答案 token 不得超过 MAX_ANSWER_TOKENS
            if len(tokenizer.tokenize(answer_text)) > MAX_ANSWER_TOKENS:
                skipped_count += 1
                continue

            # ✅ 3. 答案不能出现在 context 最后一段（最后 15 个字符内）
            if answer_start >= len(context) - 15:
                skipped_count += 1
                continue

            # ✅ 如果都符合条件，保留
            qas.append(qa)
            kept_count += 1

        if qas:
            cleaned_paragraphs.append({
                "context": context,
                "qas": qas
            })

    if cleaned_paragraphs:
        cleaned_data["data"].append({
            "title": title,
            "paragraphs": cleaned_paragraphs
        })

# ✅ 写入清洗后的新文件
with open("data/squad/clean_train.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

print(f"✅ Done! Kept {kept_count} QAs, skipped {skipped_count}. Cleaned file saved to data/squad/clean_train.json")
