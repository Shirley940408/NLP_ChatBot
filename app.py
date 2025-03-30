from flask import Flask, request, render_template
from scripts.qa_retrieval import qa_with_auto_context
import re
import textwrap

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    context = ""
    
    if request.method == "POST":
        question = request.form.get("question")
        result = qa_with_auto_context(question)

        # 原始 context
        raw_context = result["context"]

        # 1. 去掉所有节号（任意位置的数字+空格或字母），如 "25And", "3. Then"
        cleaned = re.sub(r"\b\d+\s*", "", raw_context)

        # 2. 去掉所有中括号内容（如 [8]、[note]、[]），包括内容
        cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)

        # 3. 替换破折号 — 为正常短横 -
        cleaned = cleaned.replace("—", " - ")

        # 4. 移除所有换行符（先合成完整句子）
        cleaned = cleaned.replace("\n", " ")

        # 5. 每个句号、问号、感叹号后自动换行（句子级）
        cleaned = re.sub(r"([.!?])\s+", r"\1\n", cleaned)

        # 6. 去除多余空格
        cleaned = cleaned.strip()

        # 7. 最终按 90 字宽度 wrap，每行不太长
        wrapped_context = "\n".join(textwrap.wrap(cleaned, width=90))

        answer = result["answer"]
        context = wrapped_context

    return render_template("index.html", answer=answer, context=context)

if __name__ == "__main__":
    app.run(debug=True)
