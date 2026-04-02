"""
Interface of the FinDER dataset.

FinDER (Financial Dataset for Question Answering and Evaluating Retrieval-Augmented
Generation) is a specialized benchmark for financial question answering and RAG systems.

Dataset Source: https://huggingface.co/datasets/Linq-AI-Research/FinDER

Description:
    A financial QA dataset containing 5,703 expert-annotated query-evidence-answer
    triplets sourced from real-world 10-K filings and ambiguous financial queries
    from industry professionals. The dataset emphasizes domain-specific challenges
    such as handling short, acronym-heavy queries and retrieving precise information
    from lengthy financial documents.

Size:
    - Total: ~5,703 examples (~13.1 MB)

Splits:
    - train: Training set
    - validation: Validation/dev set
    - test: Test set

Features:
    - query: The financial question/query (often short, with abbreviations/acronyms)
    - evidence: Retrieved evidence/passages from financial documents
    - answer: The answer to the query
    - metadata: Additional information about the query-evidence pair

License: Unknown - Please check the dataset page for license information.

Language: English

Paper: "FinDER: Financial Dataset for Question Answering and Evaluating
        Retrieval-Augmented Generation" (arXiv 2025)
        https://arxiv.org/abs/2504.15800
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class FinDERDataset(VisualTextBase):
    """A consistent interface for the FinDER dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract query and answer
        query = sample["query"]
        answer = sample["answer"]

        # Build the question with evidence if available
        evidence = sample["evidence"]
        if evidence:
            question = f"Evidence: {evidence}\n\nQuestion: {query}"
        else:
            question = query

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
