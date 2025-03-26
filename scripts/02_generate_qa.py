# scripts/02_generate_qa.py
import os
import requests
import json

INPUT_FOLDER = "data/raw"
OUTPUT_FILE = "data/squad/train.json"
MODEL = "tinyllama"  # This is a lightweight one is still to heavy. 5.5GiB and i have 2.8

data = {"data": []}

def generate_qa(text_chunk):
    prompt = f"""
    You are a divine and wise guide.

    Given the Bible passage below, generate one meaningful question a human might ask based on it, 
    and provide a short answer using exact words or phrases from the passage.

    Respond in this format:
    Question: <your question>
    Answer: <a direct quote from the passage>

    Passage:
    {text_chunk.strip()}
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False}
    ).json()
    return response.get("response", "").strip()

def build_squad_record(context, model_response, idx):
    try:
        # Split into lines and clean up
        lines = [line.strip() for line in model_response.strip().split("\n") if line.strip()]
        question = ""
        answer = ""

        for line in lines:
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
            elif line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()

        if not question or not answer:
            print(f"Incomplete QA pair. Response was:\n{model_response}")
            return None

        # Try to find answer in context (case insensitive)
        lowered_context = context.lower()
        lowered_answer = answer.lower()

        start_idx = lowered_context.find(lowered_answer)
        if start_idx == -1:
            print(f"Answer not found in context. Answer: '{answer}'")
            return None

        return {
            "context": context,
            "qas": [{
                "id": f"q{idx}",
                "question": question,
                "answers": [{"text": answer, "answer_start": start_idx}],
                "is_impossible": False
            }]
        }

    except Exception as e:
        print(f"Error building record: {e}")
        print("Model response was:\n", model_response)
        return None


idx = 0
for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".txt"):
        with open(os.path.join(INPUT_FOLDER, fname), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]

            for chunk in chunks[:30]:
                if len(chunk.strip()) < 100:
                    print("Skipping empty/short chunk")
                    continue

                print("Sending to model:", chunk[:80].replace("\n", " ") + "...")
                qa = generate_qa(chunk)
                print("Model response:", qa)

                record = build_squad_record(chunk, qa, idx)
                if record:
                    data["data"].append({"title": fname, "paragraphs": [record]})
                    idx += 1
                else:
                    print("Could not build a valid QA record")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Generated {idx} QA pairs.")

