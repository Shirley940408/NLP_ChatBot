# scripts/run_eval.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset, Dataset
from evaluate import load as load_metric
import torch
from tqdm import tqdm

# 模型路径
model_path = "models/distilbert-qa"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 加载原始 SQuAD 格式数据并扁平化
raw_dataset = load_dataset("json", data_files="data/squad/clean_train.json", field="data")["train"]

samples = []
for item in raw_dataset:
    for para in item["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]
            samples.append({
                "context": context,
                "question": question,
                "answer_text": answer["text"],
                "answer_start": answer["answer_start"]
            })

# 构建验证集（随机 10%）
dataset = Dataset.from_list(samples).train_test_split(test_size=0.1)["test"]

# 加载评估指标
metric = load_metric("squad")

# 推理函数
def get_answer(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip()

# 预测 + 对比真实答案
predictions = []
references = []

print("🧪 正在评估模型...")
for i, example in enumerate(tqdm(dataset)):
    pred = get_answer(example)
    ref = {
        "id": str(i),
        "answers": {
            "answer_start": [example["answer_start"]],
            "text": [example["answer_text"]]
        }
    }
    predictions.append({"id": str(i), "prediction_text": pred})
    references.append(ref)

# 计算分数
results = metric.compute(predictions=predictions, references=references)
print("\n✅ Evaluation Result:")
print(f"📌 Exact Match: {results['exact_match']:.2f}")
print(f"📌 F1 Score:    {results['f1']:.2f}")