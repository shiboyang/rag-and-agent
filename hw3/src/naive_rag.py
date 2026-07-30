import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

load_dotenv(r"../../.env")


# api_key = os.getenv("API_KEY")


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    context_str: str
    answer: str


class NaiveRAGSystem:
    def __init__(self, dataset_dir, dbpath, api_key):
        self.dataset_dir = dataset_dir
        self.dbpath = dbpath
        self.api_key = api_key

        self.embeddings = self._init_embedding()
        self.vectorstore = self._build_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = self._build_llm()
        self.rag_chain = self._build_workflow()

    def _init_embedding(self):
        embed = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=self.api_key)
        return embed

    def _load_pdf_docs(self, pdf_dir):
        dataset = DirectoryLoader(
            path=pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = dataset.load()
        return docs

    def _split_docs(self, docs):
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_spliter.split_documents(docs)
        result_docs = []
        for doc in split_docs:
            if hasattr(doc, "page_content") and isinstance(doc.page_content, str) and doc.page_content.strip():
                result_docs.append(doc)

        return result_docs

    def _build_vectorstore(self):
        if not os.path.exists(self.dbpath):
            docs = self._load_pdf_docs(self.dataset_dir)
            split_docs = self._split_docs(docs)
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.dbpath
            )
        else:
            vectorstore = Chroma(
                persist_directory=self.dbpath,
                embedding_function=self.embeddings
            )
        return vectorstore

    def _build_llm(self):
        return ChatOpenAI(
            model="qwen-turbo",
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _retriever_node(self, state: RAGState) -> RAGState:
        query = state["query"]
        docs = self.retriever.invoke(query)
        state["retrieved_docs"] = docs
        return state

    def _formate_context(self, state: RAGState) -> RAGState:
        docs = state["retrieved_docs"]
        context_str = "\n\n".join(doc.page_content for doc in docs)
        state["context_str"] = context_str
        return state

    def _generate_answer_node(self, state: RAGState):
        query = state["query"]
        context_str = state["context_str"]
        system_prompt = """
        你是一个AI助手，根据上下文回答用户的问题。
        要求：
        1. 严格按照上下文内容回答用户的问题
        2. 如果在上下文中没有找到有用的信息，回答用户：根据现在资料无法回答
        3. 回答的内容要简洁 准确 专业
        4. 不要编造上下文中不存在的信息回答给用户    
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "上下文：{context_str},\n用户的问题：{query}")
        ])
        message = prompt_template.format(context_str=context_str, query=query)
        ai_message = self.llm.invoke(message)
        state["answer"] = str(ai_message.content)
        return state

    def _build_workflow(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("format_context", self._formate_context)
        workflow.add_node("generate_answer", self._generate_answer_node)

        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "format_context")
        workflow.add_edge("format_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        return workflow.compile()

    def query(self, question: str):
        result = self.rag_chain.invoke({"query": question})
        return result


if __name__ == '__main__':
    dataset_path = "/home/shiby/code/rag-and-agent/hw3/data"
    db_path = "./pdf.db"
    api_key = os.getenv("API_KEY")
    rag_sys = NaiveRAGSystem(dataset_path, db_path, api_key)
    response = rag_sys.query("你好，什么是人头姿态估计")
    print(f"question: {response["query"]}")
    print(f"answer: {response["answer"]}")
