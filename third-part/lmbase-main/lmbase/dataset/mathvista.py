"""
Interface of the MathVista dataset.

The dataset is from https://huggingface.co/datasets/AI4Math/MathVista
It contains mathematical reasoning tasks with visual elements.
Splits: testmini, test
"""

import os
import re
import ast
import logging

from datasets import load_dataset

from lmbase.dataset.base import VisualTextSample, VisualTextBase


class MathVistaDataset(VisualTextBase):
    """A consistent interface for the MathVista dataset."""

    def __init__(
        self,
        split: str = "testmini",
        hf_dataname: str = None,
        config: dict = None,
    ):
        self.data_path = config["data_path"]
        self.image_path = os.path.join(self.data_path, "images")
        os.makedirs(self.image_path, exist_ok=True)

        super().__init__(split=split, hf_dataname=hf_dataname, config=config)

    def map_dataset(self):
        """Map the dataset to the desired format."""

        if self.hf_dataset is None:
            self.hf_dataset = load_dataset(self.hf_dataname, split=self.split)

        logging.info(
            "   - Mapping samples to lmbase format, i.e., lmbase.dataset.base.VisualTextBase"
        )

        # Check if hf_dataset is a DatasetDict (contains multiple splits)
        if hasattr(self.hf_dataset, "keys"):
            # Process only the current split
            if self.split in self.hf_dataset:
                split_dataset = self.hf_dataset[self.split]
                column_names = split_dataset.column_names

                self.hf_dataset = split_dataset.map(
                    self.batch_format,
                    batched=True,
                    batch_size=1000,
                    load_from_cache_file=True,
                    remove_columns=column_names,
                )
            else:
                raise ValueError(
                    f"Split '{self.split}' not found in dataset. Available splits: {list(self.hf_dataset.keys())}"
                )
        else:
            column_names = self.hf_dataset.column_names

            self.hf_dataset = self.hf_dataset.map(
                self.batch_format,
                batched=True,
                batch_size=1000,
                load_from_cache_file=True,
                remove_columns=column_names,
            )

        # Save some demo samples to the dataset folder
        self.save_example_samples(num_samples=20)

    def to_format(self, sample: dict):
        """Convert raw sample to standardized format."""
        sample_id = sample["pid"]
        self.idx += 1

        question = sample["question"].strip()

        # Extract all <imageN> tokens
        question_images = []
        image_tokens = re.findall(r"<image\d+>", question)

        # Get image data (decoded_image has the actual PIL image)
        image_data = sample["decoded_image"]

        # If no image tokens in question but image exists, add token
        if image_data is not None and not image_tokens:
            image_token = "<image1>"
            image_tokens.append(image_token)
            question = f"{image_token} {question}"

        for token in image_tokens:
            if image_data is not None:
                # Handle single image or multiple images
                if isinstance(image_data, (list, tuple)):
                    token_index = int(token.replace("<image", "").replace(">", "")) - 1
                    if 0 <= token_index < len(image_data):
                        current_image = image_data[token_index]
                    else:
                        continue
                else:
                    token_index = int(token.replace("<image", "").replace(">", "")) - 1
                    if token_index == 0:
                        current_image = image_data
                    else:
                        logging.warning(
                            "More image tokens than available images in sample %s",
                            sample_id,
                        )
                        continue

                filename = f"MathVista-{sample_id}-{token}"
                save_path = self.save_pil_image(
                    current_image, self.image_path, filename
                )
                if save_path is not None:
                    question_images.append((token, save_path))
                else:
                    logging.warning(
                        "Failed to save image for %s in sample %s",
                        token,
                        sample_id,
                    )
            else:
                logging.warning("No image data for sample %s", sample_id)

        # Process choices/options
        choices = sample["choices"]
        if choices is not None and len(choices) > 0:
            try:
                if isinstance(choices, str):
                    choices = ast.literal_eval(choices)
                option_letters = [chr(ord("A") + i) for i in range(len(choices))]
                options_str = "\n".join(
                    [
                        f"({letter}): {opt}"
                        for letter, opt in zip(option_letters, choices)
                    ]
                )
                question = (
                    f"{question}{self.SOLUTION_FORMAT_PROMPT}\nOptions:\n{options_str}"
                )
            except Exception as e:
                logging.warning(
                    "Failed to parse choices for sample %s: %s",
                    sample_id,
                    e,
                )
                question = f"{question}{self.SOLUTION_FORMAT_PROMPT}\n"
        else:
            question = f"{question}{self.SOLUTION_FORMAT_PROMPT}\n"

        groundtruth = str(sample["answer"]).strip()

        return VisualTextSample(
            main_id=sample_id,
            split=self.split,
            question=question,
            cot_answer="",
            groundtruth=groundtruth,
            question_images=question_images,
            sample_info={
                "dataset": self.hf_dataname,
                "question_type": sample["question_type"],
                "answer_type": sample["answer_type"],
                "unit": sample["unit"],
                "precision": sample["precision"],
                "query": sample["query"],
                "image_path": sample["image"],
            },
        )
