"""
Interface of the HotpotQA dataset.

The HotpotQA dataset is a question answering dataset that requires multi-hop reasoning.

Dataset Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa

Configurations:
    - distractor: Contains ~90,447 training and 7,405 validation examples (~598.66 MB).
                  Questions are answered using 10 provided paragraphs (distractor setting).
    - fullwiki: Contains ~90,447 training, 7,405 validation, and 7,405 test examples (~645.80 MB).
                Questions require reasoning over full Wikipedia corpus.

Splits:
    - train: Training set (~90,447 examples)
    - validation: Validation/dev set (~7,405 examples)
    - test: Test set (~7,405 examples, only available in fullwiki config)

Features:
    - question: The question requiring multi-hop reasoning
    - answer: The answer to the question
    - context: Sentences and titles from Wikipedia articles
    - supporting_facts: Sentence-level annotations indicating which sentences support the answer

License: CC BY-SA 4.0
"""

from datasets import load_dataset

from lmbase.dataset.base import TextSample, VisualTextBase


class HotpotQADataset(VisualTextBase):
    """A consistent interface for the HotpotQA dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""
        subset_name = self.config["subset"]
        self.hf_dataset = load_dataset(self.hf_dataname, subset_name, split=self.split)
        super().map_dataset()

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

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
                "type": sample["type"],
                "supporting_facts": sample["supporting_facts"],
                "level": sample["level"],
            },
        )
