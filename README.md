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

### Clone the repository

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

---

## 📋 Script Execution Order

Here is the recommended order to run the scripts in this project:

| Step | Script | Description |
|------|--------|-------------|
| 1️⃣ | `01_extract_pdfs.py` | (Optional) Extracts raw text from PDF files, e.g. Bible chapters |
| 2️⃣ | `02_generate_qa.py` | Generates SQuAD-style QA pairs using T5 and QA models |
| 3️⃣ | `03_clean_data.py` | Cleans the generated QA dataset for fine-tuning |
| 4️⃣ | `04_finetune.py` | Fine-tunes DistilBERT on the cleaned QA data |
| 5️⃣ | `save_file.py` | Utility for saving results or outputs to file |
| 6️⃣ | `prepare_embeddings.py` | Encodes all contexts using sentence transformers and builds FAISS index |
| 7️⃣ | `run_eval.py` | (Optional) Evaluates fine-tuned model on validation split (EM & F1) |
| 8️⃣ | `qa_retrieval.py` | Core module for semantic retrieval + question answering |
| 9️⃣ | `app.py` | ✅ Run the web app and test your system in a browser (`localhost:5000`) |

> You only need steps 1–4 if you are generating your own dataset and fine-tuning.
> For inference only, start directly from `prepare_embeddings.py` + `qa_retrieval.py`.
```

## 🎬 Demo Video

👉 [Click here to view demo.mp4](./demo.mov)
