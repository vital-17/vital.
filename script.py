import os
import time
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
#终端输入:streamlit run script.py


# =========================
# 1. 模型配置
# =========================
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),# 终端输入:$env:DASHSCOPE_API_KEY="你的key"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# =========================
# 2. 文本切块
# =========================
def split_text(text, chunk_size=50):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# =========================
# 3. Embedding
# =========================
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v1",
        input=text
    )
    return resp.data[0].embedding


def build_embeddings(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    return np.array(embeddings).astype("float32")


# =========================
# 4. FAISS 向量索引
# =========================
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_faiss(question, index, chunks, k=3):
    question_embedding = np.array([get_embedding(question)]).astype("float32")

    distances, indices = index.search(question_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        distance = float(distances[0][i])
        similarity = 1 / (1 + distance)

        results.append({
            "chunk": chunks[idx],
            "distance": distance,
            "similarity": similarity
        })

    return results


# =========================
# 5. Prompt + LLM 调用
# =========================
def build_prompt(question, context):
    return f"""你是一个严谨的知识库问答助手。

请你只根据下面提供的资料回答问题。
如果资料中没有相关信息，请回答：未找到相关信息。

资料：
{context}

问题：
{question}

回答：
"""


def ask_llm(prompt):
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return resp.choices[0].message.content


# =========================
# 6. Streamlit 前端
# =========================
st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG 知识库问答系统")
st.markdown("上传 txt 文档，系统会基于文档内容进行检索增强问答。")

uploaded_file = st.file_uploader("请上传 txt 文件", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    chunks = split_text(text)
    st.success(f"文档上传成功，文本已切分为 {len(chunks)} 个片段。")

    with st.spinner("正在构建 Embedding 和 FAISS 向量索引..."):
        embeddings = build_embeddings(chunks)
        index = build_faiss_index(embeddings)

    question = st.text_input("请输入你的问题：")

    if st.button("开始提问") and question:
        start_time = time.time()

        results = search_faiss(question, index, chunks, k=3)
        context = "\n\n".join([r["chunk"] for r in results])

        prompt = build_prompt(question, context)
        answer = ask_llm(prompt)

        end_time = time.time()

        st.subheader("🤖 回答")
        st.write(answer)

        st.subheader("🔍 Top-K 检索结果")
        for i, result in enumerate(results, start=1):
            st.markdown(f"### Top {i}")
            st.write(f"L2 距离：{result['distance']:.4f}")
            st.write(f"相似度：{result['similarity']:.6f}")
            st.write(result["chunk"])
            st.divider()

        st.subheader("🧩 Prompt 内容")
        st.code(prompt, language="text")

        st.subheader("⏱️ 推理耗时")
        st.write(f"{end_time - start_time:.2f} 秒")
