import os
import sys
from typing import List, Dict
from dotenv import load_dotenv

# Ragas 相关导入 (Ragas 0.4.3)
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI

# 加载环境变量
load_dotenv()


class EmbeddingsWrapper:
    """Embeddings 包装器，为 Ragas 的 OpenAIEmbeddings 添加 LangChain 兼容接口

    Ragas 0.4.3 的 embedding_factory 返回的对象使用 embed_text/embed_texts 方法，
    但某些内部组件可能期望 LangChain 风格的 embed_query/embed_documents 方法。
    此包装器提供兼容性适配。
    """

    def __init__(self, base_embeddings):
        """
        初始化包装器

        Args:
            base_embeddings: Ragas 的 OpenAIEmbeddings 对象
        """
        self.base_embeddings = base_embeddings

    def embed_query(self, text: str):
        """嵌入单个查询（LangChain 兼容接口）"""
        return self.base_embeddings.embed_text(text)

    def embed_documents(self, texts: List[str]):
        """嵌入多个文档（LangChain 兼容接口）"""
        return self.base_embeddings.embed_texts(texts)

    def embed_text(self, text: str):
        """代理到原始方法"""
        return self.base_embeddings.embed_text(text)

    def embed_texts(self, texts: List[str]):
        """代理到原始方法"""
        return self.base_embeddings.embed_texts(texts)


class RagasEvaluator:
    """基于 Ragas 的 RAG 评估器

    封装 Ragas 评估框架，提供简化的接口进行 RAG 系统评估。
    支持多种评估指标的灵活组合和批量评估。
    """

    def __init__(
            self,
            model_name: str = "qwen-plus",
            embedding_model: str = "text-embedding-v3",
            api_key: str = None,
            temperature: float = 0.3
    ):
        """
        初始化评估器

        Args:
            model_name: 评估用 LLM 模型名称
            embedding_model: Embedding 模型名称
            api_key: API 密钥，默认从环境变量读取 DASHSCOPE_API_KEY
            temperature: LLM 温度参数，控制输出的随机性
        """
        if api_key is None:
            api_key = os.getenv("API_KEY")

        if not api_key:
            raise ValueError(
                "未找到 DASHSCOPE_API_KEY，请配置环境变量或在初始化时传入"
            )

        # 创建 OpenAI 客户端（兼容 DashScope）
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 使用 llm_factory 创建 LLM
        self.evaluator_llm = llm_factory(model=model_name, client=client)

        # 使用 embedding_factory 创建 Embeddings，并包装为 LangChain 兼容接口
        # Ragas 0.4.3 版本的 `embedding_factory` 返回的 `OpenAIEmbeddings` 对象使用 `embed_text()` 和 `embed_texts()` 方法，但某些内部组件（如 AnswerRelevancy 指标）期望 LangChain 风格的 `embed_query()` 和 `embed_documents()` 方法接口。为解决此兼容性问题，我们创建了 `EmbeddingsWrapper` 包装器类，提供双向接口支持。
        base_embeddings = embedding_factory(model=embedding_model, client=client)
        self.embeddings = EmbeddingsWrapper(base_embeddings)

        print(f"✓ Ragas 评估器初始化成功")
        print(f"  - 评估模型: {model_name}")
        print(f"  - Embedding模型: {embedding_model}")
        print(f"  - 温度参数: {temperature}")

    def prepare_dataset(self, data: List[Dict]) -> EvaluationDataset:
        """
        将原始数据转换为 Ragas 评估数据集格式

        Args:
            data: 原始评估数据列表，每个元素包含：
                - question/user_input: 问题
                - answer/response: RAG 生成的答案
                - context_retrieved/retrieved_contexts: 检索到的文档列表
                - ground_truth/reference: 标准答案（可选，用于 ContextRecall）

        Returns:
            Ragas EvaluationDataset 对象
        """
        formatted_data = []

        for item in data:
            # 统一字段名称
            formatted_item = {
                "user_input": item.get("question") or item.get("user_input", ""),
                "response": item.get("answer") or item.get("response", ""),
                "retrieved_contexts": (
                        item.get("context_retrieved") or
                        item.get("retrieved_contexts") or
                        []
                ),
            }

            # 添加标准答案（如果存在）
            if "ground_truth" in item or "reference" in item:
                formatted_item["reference"] = (
                        item.get("ground_truth") or item.get("reference", "")
                )

            formatted_data.append(formatted_item)

        # 创建 EvaluationDataset
        dataset = EvaluationDataset.from_list(formatted_data)

        print(f"✓ 数据集准备完成，共 {len(formatted_data)} 条样本")
        return dataset

    def evaluate_full(
            self,
            data: List[Dict],
            metrics: List = None
    ) -> dict:
        """
        执行完整评估（需要标准答案）

        适用于严谨的基准测试，需要 ground_truth 字段。

        Args:
            data: 评估数据列表（必须包含 ground_truth/reference）
            metrics: 评估指标列表，默认为全部4个核心指标

        Returns:
            评估结果字典，包含各指标的平均分数
        """
        if metrics is None:
            metrics = [
                # 忠诚度评估 (Faithfulness): 答案是否基于检索上下文
                Faithfulness(llm=self.evaluator_llm),
                # 回答相关性 (AnswerRelevancy): 答案与问题的相关程度
                AnswerRelevancy(llm=self.evaluator_llm, embeddings=self.embeddings),
                # 上下文精确度 (ContextPrecision): 检索文档的相关性和排序
                ContextPrecision(llm=self.evaluator_llm),
                # 上下文召回率 (ContextRecall): 检索覆盖度
                ContextRecall(llm=self.evaluator_llm),
            ]

        # 准备数据集
        dataset = self.prepare_dataset(data)

        # 执行评估
        print(f"\n开始完整评估（{len(metrics)} 个指标）...")

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        # 解析结果
        result_dict = self._parse_results(results)

        return result_dict

    def _parse_results(self, results) -> dict:
        """
        解析 Ragas 评估结果

        Args:
            results: Ragas EvaluationResult 对象

        Returns:
            包含平均分数和详细结果的字典
        """
        # 转换为 DataFrame
        df = results.to_pandas()

        # 提取指标列（排除 user_input, response 等非指标列）
        metric_columns = [
            col for col in df.columns
            if col not in ["user_input", "response", "retrieved_contexts", "reference"]
        ]

        # 计算平均分
        avg_scores = {}
        for col in metric_columns:
            avg_scores[col] = df[col].mean()

        # 构建结果字典
        result_dict = {
            "average_scores": avg_scores,
            "detailed_results": df,
            "num_samples": len(df),
        }

        return result_dict

    def print_report(self, result_dict: dict):
        """
        打印评估报告

        Args:
            result_dict: evaluate() 返回的结果字典
        """
        print("\n" + "=" * 70)
        print("RAGAS 评估报告")
        print("=" * 70)
        print(f"样本数量: {result_dict['num_samples']}")
        print("-" * 70)

        # 打印平均分数
        print("\n【平均分数】")
        for metric_name, score in result_dict["average_scores"].items():
            # 美化指标名称
            display_name = self._format_metric_name(metric_name)
            print(f"  {display_name}: {score:.4f}")

        # 打印详细结果
        print("\n【详细结果】")
        print(result_dict["detailed_results"].to_string(index=False))

        print("\n" + "=" * 70)

    def save_report(
            self,
            result_dict: dict,
            output_file: str = "ragas_evaluation_report.csv"
    ):
        """
        保存评估报告到 CSV 文件

        Args:
            result_dict: evaluate() 返回的结果字典
            output_file: 输出文件路径
        """
        df = result_dict["detailed_results"]
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✓ 评估报告已保存到: {output_file}")

    @staticmethod
    def _format_metric_name(metric_name: str) -> str:
        """
        格式化指标名称为中文显示

        Args:
            metric_name: 英文指标名称

        Returns:
            中文显示名称
        """
        name_mapping = {
            "faithfulness": "忠实度 (Faithfulness)",
            "answer_relevancy": "回答相关性 (Answer Relevancy)",
            "context_precision": "上下文精确度 (Context Precision)",
            "context_recall": "上下文召回率 (Context Recall)",
            "factual_correctness": "事实正确性 (Factual Correctness)",
        }
        return name_mapping.get(metric_name.lower(), metric_name)


def main():
    """主函数 - 演示 Ragas 评估器的使用"""

    # 准备测试数据
    test_data = [
        {
            'question': '非洲的猴面包树果实的长度约是多少厘米？',
            'answer': '非洲猴面包树的果实长约15至20厘米。',
            'context_retrieved': [
                '非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。',
                '钙含量比菠菜高50％以上，含较高的抗氧化成分。',
            ],
            'context_reference': [
                '非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。'
            ]
        },
        {
            'question': '什么是机器学习？',
            'answer': '机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能。',
            'context_retrieved': [
                '机器学习是人工智能的核心技术之一，通过算法让计算机系统从数据中学习模式。',
                '深度学习是机器学习的子领域，使用多层神经网络处理复杂任务。',
            ],
            'context_reference': [
                '机器学习是人工智能的核心技术之一，通过算法让计算机系统从数据中学习模式。'
            ]
        }
    ]

    print("=" * 70)
    print("基于 Ragas 库的 RAG 评估示例")
    print("=" * 70)
    print()

    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")

    try:
        # 初始化评估器
        evaluator = RagasEvaluator(model_name="qwen-plus")

        # 测试 : 完整评估（需要标准答案）
        print("\n" + "=" * 70)
        print("【测试 2】完整评估 - 4个核心指标")
        print("=" * 70)
        result_full = evaluator.evaluate_full(test_data)
        evaluator.print_report(result_full)

        # 保存结果
        evaluator.save_report(result_full, "ragas_full_results.csv")

        print("评估完成！")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
