import os
from collections import defaultdict
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from sentence_transformers import CrossEncoder

load_dotenv(r"../../.env")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'


class OfflineRerankComponent:
    def __init__(self, top_n: int = 6, model_name: str = "BAAI/bge-reranker-base"):
        self.top_n = top_n
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document]):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model(pairs)
        socred_docs = list(zip(docs, scores))
        socred_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in socred_docs[:self.top_n]]
        return reranked_docs


class LLMComponent:
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("API_KEY"),
            temperature=0.7
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是智能助手，请根据上下文回答问题\n\n 上下文{context}\n\n, 在返回的问题中同时返回参考文档内容，并说明你的回答参考了哪些内容，说明为什么这样回答"),
            ("human", "{question}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, context, question):
        return self.chain.invoke({"context": context, "question": question})


class DocumentStoreComponent:
    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=os.getenv("API_KEY"))
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=self.embedding
        )


class RerankRetrieverComponent:
    def __init__(self, vector_store: Chroma, top_k: int = 12, top_n: int = 6):
        self.similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        self.reranker = OfflineRerankComponent(top_n)

    def retrieve(self, query):
        docs = self.similarity_retriever.invoke(query)
        reranked_docs = self.reranker.rerank(query, docs)
        return reranked_docs


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    retrieved_docs: List[Document]


class RRFRAGPipeline:
    def __init__(self, top_k: int = 6, rrf_k: int = 60):
        self.doc_store = DocumentStoreComponent("/home/shiby/code/rag-and-agent/hw4/data/db", "test_collect")
        self.retriever = RerankRetrieverComponent(self.doc_store.vector_store)
        self.llm = LLMComponent("qwen-turbo")
        self.graph = self.create_rag_graph()

    def retriever_node(self, state: RAGState):
        question = state["question"]
        retrieved_docs = self.retriever.retrieve(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        state["context"] = context
        state["retrieved_docs"] = retrieved_docs
        return state

    def generate_node(self, state: RAGState):
        question = state["question"]
        context = state["context"]
        answer = self.llm.generate(context, question)
        state["answer"] = answer
        return state

    def create_rag_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generate", self.generate_node)
        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def query(self, question):
        state = RAGState(question=question, context="", answer="", retrieved_docs=list())
        state = self.graph.invoke(state)
        return state


def create_gradio_app(pipeline: RRFRAGPipeline):
    import gradio as gr
    def chat(message, history):
        if not message:
            return "", history
        result = pipeline.query(message)
        answer = result["answer"]

        new_history = list(history) if history else []
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": answer})
        return "", new_history

    with gr.Blocks(title="多路召回+RRF问答工作流") as app:
        chat_bot = gr.Chatbot(height=500)
        with gr.Row():
            msg = gr.Textbox("请输入问题", placeholder="输入问题", scale=4)
            submit_btn = gr.Button("提交", variant="primary", scale=1)
        clear_btn = gr.Button("清空对话")

        submit_btn.click(fn=chat, inputs=[msg, chat_bot], outputs=[msg, chat_bot])
        msg.submit(fn=chat, inputs=[msg, chat_bot], outputs=[msg, chat_bot])
        clear_btn.click(fn=lambda: [], inputs=None, outputs=chat_bot, queue=False)
    return app


if __name__ == "__main__":
    rag_pipeline = RRFRAGPipeline()
    app = create_gradio_app(rag_pipeline)
    app.launch()
