"""
Interface of the ConcurrentQA dataset.

Dataset Source: https://huggingface.co/datasets/stanfordnlp/concurrentqa-retrieval

Description:
    A question-answering dataset designed for concurrent retrieval across multiple
    data sources (Wikipedia and emails). Useful for studying generalization and
    privacy in retrieval systems.

Size: ~18,400 rows total

Splits:
    - train: 15,200 examples
    - validation: 1,600 examples
    - test: 1,600 examples

Features:
    - _id: Unique identifier
    - question: The question text
    - answers: List of possible answers
    - pos_paras: Positive paragraphs (relevant documents)
    - neg_paras: Negative paragraphs (irrelevant documents)

License: MIT
Language: English
"""

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class ConcurrentQADataset(VisualTextBase):

    def to_format(self, sample):

        question = sample["question"]
        answer = sample["answers"][0]

        pos_docs = sample["pos_paras"]
        neg_docs = sample["neg_paras"]

        docs = pos_docs + neg_docs

        context = ""

        for i, d in enumerate(docs):
            context += f"\nDocument {i+1}: {d['title']}\n{d['text']}\n"

        question_with_context = f"""
Answer the question using the following documents.

Context:
{context}

Question:
{question}
"""

        return VisualTextSample(
            question=question_with_context,
            groundtruth=answer,
            cot_answer=answer,
            main_id=sample["_id"],
            split=self.split,
            sample_info=sample,
        )
