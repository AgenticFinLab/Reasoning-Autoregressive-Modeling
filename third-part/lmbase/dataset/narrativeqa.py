"""
Interface of the NarrativeQA dataset.

NarrativeQA is a reading comprehension dataset designed to evaluate deep understanding
of long-form narratives such as books and movie scripts.

Dataset Source: https://huggingface.co/datasets/meithnav/narrativeqa
(Original: https://huggingface.co/datasets/deepmind/narrativeqa)

Description:
    A comprehensive reading comprehension dataset containing approximately 46,765
    question-answer pairs derived from 1,567 documents (books and movie scripts).
    Unlike traditional QA datasets, NarrativeQA emphasizes integrative reasoning
    across entire stories, requiring models to synthesize information from dispersed
    narrative segments rather than relying on surface-level cues. Documents average
    around 60,000 tokens, presenting significant challenges for long-context understanding.

Size:
    - Documents: ~1,567 stories/screenplays
    - Question-answer pairs: ~46,765
    - Training set: ~32,747 examples
    - Validation set: ~3,461 examples
    - Test set: ~10,557 examples

Splits:
    - train: Training set (~32,747 examples)
    - validation: Validation/dev set (~3,461 examples)
    - test: Test set (~10,557 examples)

Features:
    - document: The full story text (book or movie screenplay)
    - summary: Wikipedia summary of the document
    - question: Human-written question about the narrative
    - answer: Human-written answer (multiple answers possible)
    - document_id: Unique identifier for the document

License: Apache-2.0

Language: English

Paper: "The NarrativeQA Reading Comprehension Challenge" (TACL 2018)
        https://arxiv.org/abs/1712.07040
"""

from lmbase.dataset.base import TextSample, VisualTextBase


class NarrativeQADataset(VisualTextBase):
    """A consistent interface for the NarrativeQA dataset."""

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract question and answer
        question = sample["question"]
        answer = sample["answer"]

        # Build the question with document context if available
        document = sample["document"]
        if document:
            question = f"Document: {document}\n\nQuestion: {question}"

        question = f"{question}{self.SOLUTION_FORMAT_PROMPT}"

        # Get summary if available
        summary = sample["summary"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
                "summary": summary,
            },
        )
