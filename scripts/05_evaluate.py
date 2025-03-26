# scripts/05_evaluate.py

results = [
    {"question": "What did God create?", "answer": "the heavens and the earth", "score": 1},
    {"question": "Who created everything?", "answer": "God", "score": 1},
    {"question": "Where was God?", "answer": "Unknown", "score": 0.5},
    {"question": "What happened to Earth?", "answer": "It was void", "score": 0},
]

accuracy = sum(r["score"] for r in results) / len(results)
print(f"📊 Human-rated accuracy: {accuracy:.2f}")
