import os
import sys
import json
from typing import List, Dict
from dotenv import load_dotenv

# 根据自己工程环境调整路径加载
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
# sys.path.insert(0, os.path.dirname(__file__))

from ragas_evaluator import RagasEvaluator
from naive_rag import NaiveRAGSystem

load_dotenv(r"../../.env")


class RAGEvaluator:
    """基于 Ragas 的 RAG 系统评估器"""

    def __init__(self, model_name: str = "qwen-plus"):
        if not os.getenv("API_KEY"):
            raise ValueError("未找到环境变量 DASHSCOPE_API_KEY")

        self.ragas_evaluator = RagasEvaluator(model_name=model_name)

        print(f"✓ RAG 评估器初始化成功")
        print(f"  - 评估模型: {model_name}")

    def evaluate_all(self, results: List[Dict]) -> Dict:
        """执行完整评估"""
        print("=" * 70)
        print("开始 RAG 系统评估（基于 Ragas 框架）")
        print("=" * 70)

        ragas_data = []
        for item in results:
            ragas_item = {
                'question': item['query'],
                'answer': item['answer'],
                'context_retrieved': item['context_retrieved'],
            }

            if item.get('ground_truth'):
                ragas_item['ground_truth'] = item['ground_truth']
            elif item.get('context_reference'):
                ragas_item['ground_truth'] = item['context_reference'][0] if item['context_reference'] else ''

            ragas_data.append(ragas_item)

        print("\n执行完整评估（4个核心指标）...")
        eval_result = self.ragas_evaluator.evaluate_full(ragas_data)

        self.ragas_evaluator.print_report(eval_result)

        csv_path = "/home/shiby/code/rag-and-agent/hw3/data/ragas_evaluation_results.csv"
        self.ragas_evaluator.save_report(eval_result, csv_path)
        print(f"\n✓ 详细结果已保存到: {csv_path}")

        return eval_result

    def generate_report(self, eval_result: Dict, results: List[Dict]) -> str:
        """
        生成评估报告

        Args:
            eval_result: 评估结果字典
            results: 原始测试结果列表

        Returns:
            Markdown 格式的评估报告
        """
        avg_scores = eval_result['average_scores']
        detailed_df = eval_result['detailed_results']
        num_samples = eval_result['num_samples']

        report = f"""# RAG 系统评估报告（基于 Ragas 框架）

## 评估概览
- **测试样本数量**: {num_samples}
- **评估模型**: qwen-plus
- **评估框架**: Ragas v0.4.3

## 评估指标得分

| 指标 | 平均分 | 说明 |
|------|--------|------|
"""

        metric_descriptions = {
            'faithfulness': '忠实度 - 答案是否基于检索上下文，无幻觉',
            'answer_relevancy': '回答相关性 - 答案与问题的相关程度',
            'context_precision': '上下文精确度 - 检索文档的相关性和排序质量',
            'context_recall': '上下文召回率 - 检索覆盖度'
        }

        for metric_name, score in avg_scores.items():
            display_name = metric_descriptions.get(metric_name, metric_name)
            report += f"| {display_name} | {score:.4f} | |\n"

        report += f"""

## 指标详细说明

### 1. Faithfulness (忠实度)
衡量生成的答案是否完全基于检索到的上下文，不包含幻觉或编造的内容。
- **评分范围**: 0-1，越高越好
- **优化方向**: 提高检索质量，优化提示词减少幻觉

### 2. AnswerRelevancy (回答相关性)
评估生成的答案与用户问题的相关程度，判断答案是否切题。
- **评分范围**: 0-1，越高越好
- **优化方向**: 改进提示词设计，确保答案直接回应问题

### 3. ContextPrecision (上下文精确度)
评估检索到的文档与问题的相关程度以及排序质量。
- **评分范围**: 0-1，越高越好
- **优化方向**: 优化向量检索、引入重排序模型

### 4. ContextRecall (上下文召回率)
评估检索是否覆盖了回答问题所需的所有关键信息。
- **评分范围**: 0-1，越高越好
- **优化方向**: 增加 top_k 参数、改进索引策略

## 分数解读

| 分数范围 | 评价 | 建议 |
|---------|------|------|
| 0.8 - 1.0 | 优秀 | 保持当前策略 |
| 0.6 - 0.8 | 良好 | 微调参数可进一步提升 |
| 0.4 - 0.6 | 一般 | 需要针对性优化 |
| < 0.4 | 较差 | 需要大幅改进 |

## 详细评估结果

"""

        for i, row in detailed_df.iterrows():
            report += f"### 样本 {i + 1}\n\n"
            report += f"**问题**: {row.get('user_input', '')}\n\n"
            report += f"**答案**: {row.get('response', '')[:200]}...\n\n"

            for metric_name in avg_scores.keys():
                if metric_name in row.index:
                    score = row[metric_name]
                    report += f"- {metric_name}: {score:.4f}\n"

            report += "\n---\n\n"

        report += """
## 优化建议

根据评估结果，建议采取以下优化措施：

1. **提高忠实度**
   - 优化提示词，明确要求答案必须基于检索内容
   - 添加引用标注，让模型明确指出信息来源
   - 使用更强大的 LLM 模型

2. **提高回答相关性**
   - 改进提示词工程，使用思维链引导
   - 在提示词中强调直接回答问题
   - 调整温度参数降低随机性

3. **提高上下文精确度**
   - 优化文本分割策略（块大小、重叠）
   - 引入重排序模型（如 bge-reranker）
   - 调整向量检索的相似度阈值

4. **提高上下文召回率**
   - 增加检索返回的文档数量（top_k）
   - 使用混合检索（向量 + 关键词）
   - 改进索引策略（多字段索引）

## 技术说明

- **评估框架**: [Ragas](https://docs.ragas.io/) v0.4.3
- **评估方法**: LLM-as-Judge（使用 qwen-plus 作为裁判模型）
- **数据来源**: `data/test_results.json`
- **详细结果**: `data/ragas_evaluation_results.csv`

---

*本报告由 Ragas 评估框架自动生成*
"""

        return report


def collect_rag_results(test_data_path="data/test.json", output_path="data/test_results.json"):
    """
    采集 RAG 系统输出

    Args:
        test_data_path: 测试数据文件路径
        output_path: 输出结果文件路径

    Returns:
        测试结果列表
    """
    print("=" * 70)
    print("采集 RAG 系统输出")
    print("=" * 70)

    rag_system = NaiveRAGSystem(
        dataset_dir="/home/shiby/code/rag-and-agent/hw3/data",
        dbpath="/home/shiby/code/rag-and-agent/hw3/src/pdf.db",
        api_key=os.getenv("API_KEY")
    )

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\n加载测试数据: {len(test_data)} 条样本\n")

    results = []
    for i, item in enumerate(test_data):
        print(f"处理第 {i + 1}/{len(test_data)} 条测试数据...")
        try:
            result = rag_system.query(item["query"])

            results.append({
                "query": item["query"],
                "answer": result["answer"],
                "context_retrieved": [doc.page_content for doc in result["retrieved_docs"]],
                "context_reference": item.get("reference", []),
                "ground_truth": item.get("ground_truth", "")
            })
            print(f"  ✓ 完成")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 结果已保存到 {output_path}")
    print(f"✓ 成功处理 {len(results)}/{len(test_data)} 条样本")

    return results


def run_evaluation(results_path:str):
    """运行完整评估流程"""
    evaluator = RAGEvaluator()

    print(f"\n加载测试结果: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"加载 {len(results)} 条测试结果")

    if not results:
        print("\n错误: 没有可用的测试结果")
        return

    eval_result = evaluator.evaluate_all(results)

    print("\n生成评估报告...")
    report = evaluator.generate_report(eval_result, results)

    report_path = "/home/shiby/code/rag-and-agent/hw3/data/evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ 报告已保存到 {report_path}")


if __name__ == "__main__":
    # collect_rag_results(
    #     test_data_path="/home/shiby/code/rag-and-agent/hw3/data/qa_dataset.json",
    #     output_path="/home/shiby/code/rag-and-agent/hw3/data/qa_dataset_result.json"
    # )
    run_evaluation(results_path="/home/shiby/code/rag-and-agent/hw3/data/qa_dataset_result.json")
