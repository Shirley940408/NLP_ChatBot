# 📖 Automated Question Answering System

A simple but complete extractive QA system that allows users to ask natural language questions. The system retrieves the most relevant passage from a dataset (e.g. Bible) and uses a fine-tuned DistilBERT model to extract the best answer.

This project integrates:
- 🤗 Hugging Face Transformers for model loading and inference
- 📚 Semantic search with Sentence-Transformers + FAISS
- 🌐 A clean Flask-based web interface
- ✅ Optional model fine-tuning with SQuAD-style dataset

---

## 🚀 Features

- Input a natural language question
- Retrieve best-matching context using semantic search
- Answer the question using a QA model
- Clean, user-friendly web interface

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/qa-bot.git
cd qa-bot

## Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
pip install PyMuPDF
pip install datasets
pip install tf-keras
pip install spacy
python -m spacy download en_core_web_sm