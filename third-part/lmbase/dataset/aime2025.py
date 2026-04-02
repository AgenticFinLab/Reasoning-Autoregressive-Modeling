"""
Interface of the AIME2025 dataset.

Dataset Source: https://huggingface.co/datasets/opencompass/AIME2025

Description:
    Math problems from the 2025 American Invitational Mathematics Examination (AIME).
    Contains problems from both AIME I and AIME II (15 problems each).

Size: ~10.9k tokens, ~30 samples

Configurations:
    - AIME2025-I: First AIME exam (15 problems)
    - AIME2025-II: Second AIME exam (15 problems)
    Config setting in code: subset="AIME2025-I" or subset="AIME2025-II"

Splits:
    - test: Test set (15 examples per configuration)

Features:
    - question: The math problem statement
    - answer: Final answer

License: MIT
Language: English
"""

from datasets import load_dataset, config as hf_config
from lmbase.dataset.base import TextSample, VisualTextBase
import logging


class AIME2025Dataset(VisualTextBase):
    """A consistent interface for the AIME2025 dataset."""

    def __init__(self, split="test", **kwargs):
        config = kwargs.get("config", {})
        if "subset" not in config:
            raise ValueError(
                "Config must contain 'subset' key for AIME2025Dataset, "
                "specifying 'AIME2025-I' or 'AIME2025-II'"
            )
        subset = config["subset"]
        if subset not in ["AIME2025-I", "AIME2025-II"]:
            raise ValueError(
                f"Subset must be 'AIME2025-I' or 'AIME2025-II', got {subset}"
            )
        self.subset = subset
        super().__init__(split=split, **kwargs)

    def map_dataset(self):
        """
        Map HF dataset rows to standardized sample records.
        """
        if self.hf_dataset is None:
            logging.info("   - HF cache directory: %s", hf_config.HF_DATASETS_CACHE)
            # Use self.subset as the subset configuration, and self.split as the HF split
            self.hf_dataset = load_dataset(
                self.hf_dataname,
                self.subset,
                split=self.split,
            )

        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.TextSample"
        )
        print(self.hf_dataset.column_names)
        self.hf_dataset = self.hf_dataset.map(
            self.batch_format,
            batched=True,
            batch_size=1000,
            load_from_cache_file=True,
            remove_columns=self.hf_dataset.column_names,
        )
        # We need smaller number of samples because the AIME25 only
        # have 15 samples in each split.
        self.save_example_samples(num_samples=5)

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        problem = sample["question"]
        question = f"{problem}{self.SOLUTION_FORMAT_PROMPT}"
        answer = sample["answer"]

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer,
            groundtruth=answer,
            sample_info={
                "dataset": self.hf_dataname,
                "subset": self.subset,
            },
        )
