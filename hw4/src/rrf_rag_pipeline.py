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

load_dotenv(r"../../.env")


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


class MultiPathRetriever:
    def __init__(self, vector_store: Chroma, top_k: int = 6):
        self.mp_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k * 2,
                "lambda_mult": 0.5
            }
        )
        self.sim_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k
            }
        )

    def retrieve(self, query):
        mpr_docs = self.mp_retriever.invoke(query)
        sim_docs = self.sim_retriever.invoke(query)
        return sim_docs, mpr_docs


class RRFusionComponent:
    def __init__(self, k: int):
        self.k = k

    def rrfusion(self, retrieved_docs):
        scores = defaultdict(float)
        doc_map = {}

        for docs in retrieved_docs:
            for rank, doc in enumerate(docs, start=1):
                doc_id = doc.metadata["id"]
                scores[doc_id] = 1.0 / (self.k + rank)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        result = sorted(doc_map, key=lambda x: -x[1])
        return result

    def refusion_docs(self, retrieved_docs):
        sorted_docs = self.rrfusion(retrieved_docs)
        k_docs = []
        for doc, score in sorted_docs[:self.k]:
            k_docs.append(Document(page_content=doc.page_content))

        return k_docs


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    retrieved_docs: List[Document]


class RRFRAGPipeline:
    def __init__(self, top_k: int = 6, rrf_k: int = 60):
        self.doc_store = DocumentStoreComponent()
        self.retriever = MultiPathRetriever(self.doc_store.vector_store)
        self.rrfusino = RRFusionComponent(rrf_k)
        self.llm = LLMComponent()
        self.graph = self.create_rag_graph()

    def retriever_node(self, state: RAGState):
        question = state["question"]
        retrieved_docs = self.retriever.retrieve(question)
        docs = self.rrfusino.refusion_docs(retrieved_docs)
        context = "\n\n".join([doc.page_content for doc in docs])

        state["context"] = context
        state["retrieved_docs"] = docs
        return state

    def generate_node(self, state: RAGState):
        question = state["question"]
        context = state["context"]
        answer = self.llm.generate(context, question)
        state["answer"] = answer

    def create_rag_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_edge("retriever", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def query(self, question):
        state = RAGState()
        self.graph.invoke(question)
