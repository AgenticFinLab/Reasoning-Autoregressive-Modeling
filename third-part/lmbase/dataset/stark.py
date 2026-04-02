"""
Interface of the STaRK dataset.

STaRK (Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases) is a
large-scale benchmark for evaluating retrieval capabilities of LLMs on semi-structured
knowledge bases across diverse domains.

Dataset Source: https://huggingface.co/datasets/snap-stanford/stark
Documentation: https://stark.stanford.edu/

Description:
    A benchmark for evaluating LLM retrieval on textual and relational knowledge bases.
    It includes three knowledge bases across different domains:
    - Amazon: Product search knowledge base
    - MAG (Microsoft Academic Graph): Academic research knowledge base
    - Prime: Biomedical knowledge base

Size:
    - Amazon: ~XX,XXX queries (synthesized + human-generated)
    - MAG: ~XX,XXX queries (synthesized + human-generated)
    - Prime: ~XX,XXX queries (synthesized + human-generated)

Configurations:
    - amazon: Product search domain
    - mag: Academic research domain
    - prime: Biomedical domain
    Config setting in code: subset="amazon" or subset="mag" or subset="prime"

Splits:
    - synthesized_all_split: Main dataset with synthesized queries (~11,204 examples for Prime)
    - humen_generated_eval: Human-generated evaluation set (~98 examples for Prime)

Features:
    - query: The natural language query/question
    - query_id: Unique identifier for the query
    - answer_ids: List of answer node IDs
    - metadata: Additional information (removed in released version to prevent leakage)

License: Unknown - Please check the dataset page for license information.

Language: English

Paper: "STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases"
        https://arxiv.org/abs/2404.13207
"""

from datasets import load_dataset

from lmbase.dataset.base import TextSample, VisualTextBase


class STaRKDataset(VisualTextBase):
    """A consistent interface for the STaRK dataset."""

    def map_dataset(self):
        """Map the dataset to the desired format."""

        subset_name = self.config["subset"]
        # Map subset name to config name
        config_map = {
            "amazon": "STaRK-Amazon",
            "mag": "STaRK-MAG",
            "prime": "STaRK-Prime",
        }
        config_name = config_map[subset_name]

        # Load dataset using HuggingFace with config
        self.hf_dataset = load_dataset(self.hf_dataname, config_name, split=self.split)
        super().map_dataset()

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Extract fields from the HuggingFace dataset format
        query = sample["query"]
        answer_ids = sample["answer_ids"]

        # Convert answer_ids to string for groundtruth
        if isinstance(answer_ids, list):
            answer = ", ".join(str(aid) for aid in answer_ids)
        else:
            answer = str(answer_ids)

        question = f"{query}{self.SOLUTION_FORMAT_PROMPT}"

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
