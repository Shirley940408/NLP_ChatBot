# scripts/qa_retrieval.py

import torch
import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# 加载模型（你 fine-tune 的 QA 模型）
model_path = "models/distilbert-qa"
qa_tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# 检索模型
encoder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("retrieval/context.index")
with open("retrieval/contexts.json", "r") as f:
    all_contexts = json.load(f)

# 主函数
def qa_with_auto_context(question, top_k=1):
    # 编码问题
    q_vec = encoder.encode([question], convert_to_numpy=True)

    # 在索引中搜索
    D, I = index.search(q_vec, top_k)
    context = all_contexts[I[0][0]]

    # QA 推理
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        "question": question,
        "context": context,
        "answer": answer.strip()
    }

# 测试：
if __name__ == "__main__":
    q = "What did God create?"
    result = qa_with_auto_context(q)
    print(f"❓ {result['question']}")
    print(f"📚 Context: {result['context'][:100]}...")
    print(f"✅ Answer: {result['answer']}")
