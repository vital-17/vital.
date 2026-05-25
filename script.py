import streamlit as st
import numpy as np
from openai import OpenAI

# 你的AI配置
client = OpenAI(
    api_key="sk-ad6dd0d9e4a044b6ae0a01ffc0a80cb7",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def split_text(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks
def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-v1",
        input=text
    )
    return resp.data[0].embedding


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
def find_most_relevant(question, chunks):
    q_emb = get_embedding(question)

    best_chunk = ""
    best_score = -1

    for chunk in chunks:
        chunk_emb = get_embedding(chunk)
        score = cosine_similarity(q_emb, chunk_emb)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk
def ask_with_rag(question, chunks):
    context = find_most_relevant(question, chunks)

    prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{question}"

    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content
text = load_text("data.txt")
chunks = split_text(text)

question = input("请输入问题：")
answer = ask_with_rag(question, chunks)

print(answer)