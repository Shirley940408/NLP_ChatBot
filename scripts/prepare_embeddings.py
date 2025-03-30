# scripts/prepare_embeddings.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# 模型：用来编码文本的 embedding 检索模型
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# 加载你的训练数据
with open("data/squad/clean_train.json", "r") as f:
    data = json.load(f)

contexts = []
for item in data["data"]:
    for para in item["paragraphs"]:
        context = para["context"]
        contexts.append(context)

# 编码所有 context
embeddings = encoder.encode(contexts, convert_to_numpy=True, show_progress_bar=True)

# 用 FAISS 构建索引
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 保存
faiss.write_index(index, "retrieval/context.index")
with open("retrieval/contexts.json", "w") as f:
    json.dump(contexts, f)

print(f"✅ 共保存 {len(contexts)} 段 context 到索引中")
