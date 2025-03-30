# scripts/02_generate_qa.py
import os
import json
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# ===== 配置参数 =====
INPUT_FOLDER = "data/raw"
OUTPUT_FILE = "data/squad/train.json"
QUESTION_GEN_MODEL = "google/flan-t5-small"
ANSWER_MODEL = "distilbert-base-cased-distilled-squad"

# ===== 加载 SpaCy 分句器 =====
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10_000_000

# ===== 加载问答模型（用于提取答案）=====
print("Loading QA model...")
qa_pipeline = pipeline("question-answering", model=ANSWER_MODEL, tokenizer=ANSWER_MODEL, device=-1)

# ===== 加载提问模型 =====
print("Loading question generation model...")
qgen_pipeline = pipeline("text2text-generation", model=QUESTION_GEN_MODEL, tokenizer=QUESTION_GEN_MODEL, device=-1)

# ===== 初始化数据结构 =====
data = {"data": []}

# ===== 生成问题的函数（用 T5）=====
def generate_question_t5(text_chunk):
    prompt = f"Generate a question from the following passage:\n\n{text_chunk.strip()}"
    try:
        response = qgen_pipeline(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        print("⚠️ Failed to generate question:", e)
        return None

# ===== 构建 SQuAD 格式记录 =====
def build_squad_record(context, question, answer_dict, idx):
    answer_text = answer_dict['answer']
    start_idx = answer_dict['start']
    if not answer_text or start_idx == -1:
        print("⚠️ Answer not found in context.")
        return None

    return {
        "context": context,
        "qas": [{
            "id": f"q{idx}",
            "question": question,
            "answers": [{"text": answer_text, "answer_start": start_idx}],
            "is_impossible": False
        }]
    }

# ===== 主流程 =====
idx = 0
for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".txt"):
        with open(os.path.join(INPUT_FOLDER, fname), "r", encoding="utf-8") as f:
            text = f.read()
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
            chunks, current = [], ""
            for sent in sentences:
                if len(current) + len(sent) < 500:
                    current += sent + " "
                else:
                    if len(current.strip()) >= 100:
                        chunks.append(current.strip())
                    current = sent + " "
            if len(current.strip()) >= 100:
                chunks.append(current.strip())

            # 限制每个文件最多生成 10 个问答对
            for i, chunk in enumerate(chunks):
                if i >= 500:
                    break
                print("📖 Generating question for:", chunk[:80].replace("\n", " ") + "...")
                question = generate_question_t5(chunk)
                if not question:
                    continue
                print("❓ Question:", question)
                try:
                    result = qa_pipeline(question=question, context=chunk)
                    print("✅ Answer:", result["answer"])
                    record = build_squad_record(chunk, question, result, idx)
                    if record:
                        data["data"].append({"title": fname, "paragraphs": [record]})
                        idx += 1
                    else:
                        print("❌ Could not build a valid QA record")
                except Exception as e:
                    print(f"❌ Failed to extract answer: {e}")

# ===== 写入输出 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"\n🎉 Done! Generated {idx} QA pairs. Output saved to {OUTPUT_FILE}")
