import os
from enum import pickle_by_enum_name
from idlelib import query
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ppt_loader import PPTXLoader

load_dotenv(r"../../.env")


class DocumentStoreComponent:
    def __init__(self, vectordb_path: str, collection_name: str):
        self.vectordb_path = vectordb_path
        self.collection_name = collection_name
        self.embed_model = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=os.getenv("API_KEY"))

        self.text_spliter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

        self.vector_store = Chroma(
            persist_directory=self.vectordb_path,
            embedding_function=self.embed_model,
            collection_name=self.collection_name
        )

        doc_count = self.vector_store._collection.count()
        print(f"向量数据库中doc的数量：{doc_count}")

    def load_documents_from_dir(self, docs_dir: str):
        docs_dir = Path(docs_dir)
        if not docs_dir.exists():
            raise FileNotFoundError(f"{docs_dir.as_posix()}")

        documents = []
        for file_path in docs_dir.iterdir():
            if file_path.is_file():
                doc = self._load_document(file_path)
                if doc:
                    documents.extend(doc)
        return documents

    def _load_document(self, filepath: Path):
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path=filepath.as_posix()).load()
        elif ext == ".pptx":
            return PPTXLoader(filepath=filepath.as_posix(), img_dir="../data/imgs").load()
        elif ext == ".txt":
            return TextLoader(file_path=filepath.as_posix()).load()
        elif ext == ".docx":
            return Docx2txtLoader(file_path=filepath.as_posix()).load()
        else:
            return None

    def add_documents(self, documents: List[Document]):
        split_docs = self.text_spliter.split_documents(documents)
        self.vector_store.add_documents(split_docs)


class RetrieveComponent:
    def __init__(self, vector_store: Chroma, top_k: int = 6):
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)


class LLMComponent:
    def __init__(self, model_name: str = "qwen-turbo"):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("API_KEY"),
            temperature=0.7
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是智能助手，请根据上下文回答问题\n\n 上下文{context}\n\n, 在返回的问题中同时返回参考文档内容，并说明你的回答参考了哪些内容，说明为什么这样回答"),
            ("human", "问题：{question}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, context: str, question: str):
        return self.chain.invoke({"context": context, "question": question})


class RAGPipeline:
    def __init__(self):
        self.doc_store = DocumentStoreComponent(
            "/home/shiby/code/rag-and-agent/hw4/data/db",
            "test_collect"
        )
        self.retriever = RetrieveComponent(self.doc_store.vector_store)
        self.llm = LLMComponent()

    def load_and_index(self, docs_dir: str):
        documents = self.doc_store.load_documents_from_dir(docs_dir)
        self.doc_store.add_documents(documents)

    def query(self, question: str):
        docs = self.retriever.retrieve(question)
        context = "\n\n".join([d.page_content for d in docs])
        answer = self.llm.generate(context, question)
        return {
            "question": question,
            "retrieved_docs": docs,
            "answer": answer
        }


def create_gradio_app(pipeline: RAGPipeline):
    import gradio as gr

    def chat(message, history):
        if not message or not message.strip():
            return "", history
        result = pipeline.query(message)
        answer = f"{result['answer']}\n\n参考{len(result['retrieved_docs'])}个文档"

        new_history = list(history) if history else []
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": answer})
        return "", new_history

    with gr.Blocks(title="RAG问答系统") as app:
        gr.Markdown("# 问答系统")
        chat_bot = gr.Chatbot(height=500)

        with gr.Row():
            msg = gr.Textbox(label="输入问题", placeholder="请输入问题", scale=4)
            submit_btn = gr.Button("发送", variant="primary", scale=1)

        clear_btn = gr.Button("清空对话")
        submit_btn.click(fn=chat, inputs=[msg, chat_bot], outputs=[msg, chat_bot])
        msg.submit(fn=chat, inputs=[msg, chat_bot], outputs=[msg, chat_bot])
        clear_btn.click(fn=lambda: [], inputs=None, outputs=chat_bot, queue=False)

    return app


if __name__ == '__main__':
    pipeline = RAGPipeline()
    pipeline.load_and_index("/home/shiby/code/rag-and-agent/hw4/data/docs")
    app = create_gradio_app(pipeline)
    app.launch()
