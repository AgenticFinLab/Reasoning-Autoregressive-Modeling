"""
Interface of the DocFinQA dataset.

DocFinQA is a long-context financial question-answering dataset designed to evaluate
large language models on extended financial documents. It extends the original FinQA
dataset by incorporating full-document contexts from SEC reports.

Dataset Source: https://huggingface.co/datasets/kensho/DocFinQA

Description:
    A financial document QA dataset with approximately 7,437 questions derived from
    SEC reports. The average context length is extended from under 700 words (FinQA)
    to approximately 123,000 words, enabling more realistic and challenging financial
    reasoning tasks over lengthy documents.

Size:
    - Total: ~7,437 questions (~4.7 GB, Parquet files ~689 MB)
    - Training set: ~4,900 examples (~3.62 GB)
    - Validation set: ~663 examples
    - Test set: ~772 examples
    - Updated test set: also available

Splits:
    - train: Training set (~4,900 examples)
    - validation: Validation/dev set (~663 examples)
    - test: Test set (~772 examples)

Features:
    - question: The financial question requiring reasoning over the document
    - answer: The numerical or textual answer to the question
    - context: Full document context from SEC reports (very long text)
    - program: Python program used to compute the answer (for numerical questions)
    - annotation: Additional metadata about the question-answer pair

License: Unknown - Please check the dataset page for license information.

Language: English

Paper: "DocFinQA: A Long-Context Financial Reasoning Dataset" (ACL 2024)
       https://arxiv.org/abs/2401.06915
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class DocFinQADataset(VisualTextBase):
    """A consistent interface for the DocFinQA dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract question and answer
        question = sample["question"]
        answer = sample["answer"]

        # Build the question with context if available
        context = sample["context"]
        if context:
            question = f"Context: {context}\n\nQuestion: {question}"

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # Get program if available (for numerical reasoning questions)
        program = sample["program"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
                "program": program,
            },
        )
