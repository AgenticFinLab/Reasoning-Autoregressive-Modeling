"""
Interface of the MultiHopRAG dataset.

Dataset Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG

Description:
    A benchmark dataset for evaluating retrieval-augmented generation on multi-hop
    queries. Designed for complex question answering requiring multi-hop reasoning
    across multiple documents.

Size: ~2,560 rows

Splits:
    - train: Training set for model training
    - corpus: Document corpus for retrieval

Features:
    - query: The multi-hop question
    - answer: The answer to the question
    - evidence_list: List of evidence documents with title, fact, and source
    - question_type: Type of question (e.g., comparison, bridge)

License: ODC-By (Open Data Commons Attribution License)
Paper: arXiv:2401.15391 "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries"
Language: English
"""

import os
from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MultiHopRAGDataset(VisualTextBase):
    """Interface for the MultiHopRAG dataset."""

    def __init__(
        self,
        split: str = "train",
        hf_dataname: str = "yixuantt/MultiHopRAG",
        config: dict = None,
    ):
        self.data_path = config.get("data_path", "./data")

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def to_format(self, sample: dict):
        """Convert HF sample to VisualTextSample format."""

        sample_id = self.idx
        self.idx += 1

        # question
        question = sample["query"].strip()

        # evidence documents
        evidence_list = sample.get("evidence_list", [])

        documents = []
        for i, ev in enumerate(evidence_list):
            title = ev.get("title", "")
            fact = ev.get("fact", "")
            source = ev.get("source", "")

            doc_text = f"Document {i+1} ({source}) - {title}: {fact}"
            documents.append(doc_text)

        context = "\n".join(documents)

        full_question = (
            f"Question:\n{question}\n\n"
            f"Documents:\n{context}\n\n"
            "Please answer the question based on the documents."
        )

        full_question = f"{full_question}{self.SOLUTION_FORMAT_PROMPT}"

        cot_answer = ""

        groundtruth = sample["answer"].strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=full_question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[],
            sample_info={
                "dataset": self.hf_dataname,
                "question_type": sample.get("question_type", ""),
            },
        )
