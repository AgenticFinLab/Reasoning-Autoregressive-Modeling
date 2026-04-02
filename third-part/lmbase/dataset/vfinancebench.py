"""
Interface of the virattt/financebench dataset.

A financial question-answering dataset focused on extracting answers from financial documents.

Dataset Source: https://huggingface.co/datasets/virattt/financebench

Description:
    A financial QA dataset containing 500 examples where each example consists of a
    question about financial data, a financial document (such as income statements,
    balance sheets, or financial reports), and the corresponding answer. The dataset
    is designed to test models' ability to extract and compute financial information
    from structured financial documents.

Size:
    - Total: 500 examples

Splits:
    - train: 500 examples (only split available)

Features:
    - question: The financial question requiring extraction or computation from the document
    - document: The financial document text (income statements, balance sheets, etc.)
    - answer: The answer to the question (numerical or textual)

License: Unknown - Please check the dataset page for license information.

Language: English
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class VFinanceBenchDataset(VisualTextBase):
    """A consistent interface for the virattt/financebench dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract question, document, and answer
        question = sample["question"]
        document = sample["document"]
        answer = sample["answer"]

        # Build the question with document context
        full_question = (
            f"Document: {document}\n\nQuestion: {question}{self.SOLUTION_FORMAT_PROMPT}"
        )

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=full_question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
            },
        )
