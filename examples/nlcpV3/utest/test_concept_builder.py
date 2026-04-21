"""Integration test for ConceptPyramidBuilder.

Loads config from YAML, initializes the real Qwen encoder from HuggingFace,
and validates the full Builder pipeline on real data.

Device is auto-detected via lmbase.utils.env_tools.get_device():
  cuda > mps (Apple Silicon) > cpu

COMMAND LINE USAGE:
  # Run with config (auto-detects best available device)
  cd /Users/sjia/Documents/AgenticFinLab/Projects/Reasoning-Autoregressive-Modeling
  python3 examples/nlcpV3/utest/test_concept_builder.py \\
      -c configs/nlcpV3/utest/test_concept_builder.yml
"""

import argparse
import logging
import sys
import unittest
from pathlib import Path

import torch

# =============================================================================
# Path Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from nlcpV3.config import NLCPV3Config
from nlcpV3.concept_hybrid_builder import (
    ConceptPyramidBuilder,
    EncoderOutput,
    PyramidOutput,
    SingleLevelOutput,
)
from lmbase.utils.env_tools import get_device
from ram.utils import load_config


# =============================================================================
# CLI Arguments
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ConceptPyramidBuilder integration test"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


# =============================================================================
# Config Builder from YAML
# =============================================================================


def build_nlcpv3_config_from_yaml(yaml_dict: dict) -> NLCPV3Config:
    """Build NLCPV3Config from parsed YAML configuration.

    Args:
        yaml_dict: Configuration dictionary loaded from YAML file

    Returns:
        NLCPV3Config instance
    """
    m = yaml_dict["model"]
    enc = m["encoder"]
    pyr = m["pyramid"]
    dec = m["decoder"]
    tr = yaml_dict["training"]
    lw = tr["loss_weights"]

    return NLCPV3Config(
        hidden_dim=pyr["hidden_dim"],
        num_heads=pyr["num_heads"],
        num_levels=pyr["num_levels"],
        level_lengths=pyr["level_lengths"],
        max_seq_len=pyr["max_seq_len"],
        encoder_model_name=enc["encoder_model_name"],
        encoder_num_layers=enc["encoder_num_layers"],
        encoder_freeze=enc["encoder_freeze"],
        vocab_size=dec["vocab_size"],
        dropout=dec["dropout"],
        rms_norm_eps=dec["rms_norm_eps"],
        muP_scale=dec["muP_scale"],
        ntp_loss_weight=lw["ntp_loss_weight"],
        concept_loss_weight=lw["concept_loss_weight"],
        recon_loss_weight=lw["recon_loss_weight"],
    )


# =============================================================================
# Main Test Class
# =============================================================================


class TestConceptPyramidBuilder(unittest.TestCase):
    """Integration test for ConceptPyramidBuilder with real Qwen model.

    All tests run with a configurable batch_size (default 3).
    Change self.batch_size in setUpClass to test with any B >= 1.
    """

    # Configurable batch size for all tests. Change to 1, 2, 4, 8, etc.
    batch_size = 3

    @classmethod
    def setUpClass(cls):
        """Load config, model, tokenizer, and dataset once for all tests."""
        cls.args = parse_args()

        # Resolve config path
        config_path = Path(cls.args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML config
        cls.yaml_config = load_config(str(config_path))
        cls.nlcp_config = build_nlcpv3_config_from_yaml(cls.yaml_config)

        # Resolve device via lmbase auto-detection (cuda > mps > cpu)
        cls.device = str(get_device("auto"))

        logging.info("=" * 60)
        logging.info("ConceptPyramidBuilder Integration Test")
        logging.info("=" * 60)
        logging.info("Encoder model: %s", cls.nlcp_config.encoder_model_name)
        logging.info(
            "Encoder layers: %s (use all if -1)", cls.nlcp_config.encoder_num_layers
        )
        logging.info("Concept dim D:  %d", cls.nlcp_config.hidden_dim)
        logging.info("Num levels K:   %d", cls.nlcp_config.num_levels)
        logging.info("Level lengths:  %s", cls.nlcp_config.level_lengths)
        logging.info("Device:         %s (auto-detected)", cls.device)

        # Load Builder with real model
        logging.info("Loading model from HuggingFace...")
        cls.builder = ConceptPyramidBuilder(cls.nlcp_config)
        cls.builder.to(cls.device)
        logging.info(
            "Model loaded. encoder_hidden_dim=%d", cls.builder.encoder_hidden_dim
        )

        # Load tokenizer
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.nlcp_config.encoder_model_name
        )
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # Load GSM8K dataset
        logging.info("Loading GSM8K dataset...")
        from lmbase.dataset import registry

        data_cfg = cls.yaml_config["data"]
        cls.dataset = registry.get(data_cfg, split="train")
        logging.info("Loaded %d GSM8K samples", len(cls.dataset))

        logging.info("=" * 60)

    # =====================================================================
    # Constructor & Config Tests
    # =====================================================================

    def test_encoder_loaded(self):
        """Verify encoder is loaded from HuggingFace with correct hidden dim."""
        self.assertIsNotNone(self.builder.encoder)
        self.assertEqual(
            self.builder.encoder_hidden_dim,
            self.builder.encoder.config.hidden_size,
        )

    def test_input_proj_dimensions(self):
        """Verify input_proj maps encoder_dim → concept_dim."""
        self.assertEqual(
            self.builder.input_proj.in_features,
            self.builder.encoder_hidden_dim,
        )
        self.assertEqual(
            self.builder.input_proj.out_features,
            self.nlcp_config.hidden_dim,
        )

    def test_concept_queries_count(self):
        """Verify correct number of query groups per level."""
        self.assertEqual(
            len(self.builder.concept_queries),
            self.nlcp_config.num_levels,
        )

    def test_concept_queries_shapes(self):
        """Verify query shapes [L_k, D] match config."""
        for level_idx, queries in enumerate(self.builder.concept_queries):
            expected_len = self.nlcp_config.level_lengths[level_idx]
            self.assertEqual(
                queries.shape,
                (expected_len, self.nlcp_config.hidden_dim),
                f"Level {level_idx} queries shape mismatch",
            )

    def test_temperature_is_learnable(self):
        """Verify temperature is a learnable scalar."""
        self.assertEqual(self.builder.temperature.shape, (1,))
        self.assertTrue(self.builder.temperature.requires_grad)

    def test_level_projs_count(self):
        """Verify correct number of level-specific projections."""
        self.assertEqual(
            len(self.builder.level_projs),
            self.nlcp_config.num_levels,
        )

    def test_recon_decoder_placeholder(self):
        """Verify recon_decoder is None (future CoT reconstruction)."""
        self.assertIsNone(self.builder.recon_decoder)

    def test_total_concepts(self):
        """Verify total concept count matches config."""
        expected = sum(self.nlcp_config.level_lengths)
        self.assertEqual(self.builder.get_total_concepts(), expected)

    # =====================================================================
    # encode_cot() Tests
    # =====================================================================

    def test_encode_cot_returns_encoder_output(self):
        """Verify encode_cot returns EncoderOutput dataclass (batch=B)."""
        texts = [
            f"Sample question {i}: What is {i} + {i+1}?" for i in range(self.batch_size)
        ]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        output = self.builder.encode_cot(input_ids)
        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(output.hidden_states.shape[0], self.batch_size)

    def test_encode_cot_shape(self):
        """Verify encoded hidden states shape [B, L, D_encoder] for any B>=1."""
        texts = [f"Query {i}: compute {i} * {i+2}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        output = self.builder.encode_cot(input_ids)
        self.assertEqual(output.hidden_states.dim(), 3)
        self.assertEqual(output.hidden_states.shape[0], self.batch_size)
        self.assertEqual(
            output.hidden_states.shape[-1],
            self.builder.encoder_hidden_dim,
        )

    def test_encode_cot_with_attention_mask(self):
        """Verify encode_cot accepts attention_mask for batch=B."""
        texts = [f"Problem {i}: Solve {i} + {i}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        output = self.builder.encode_cot(input_ids, attention_mask)
        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(output.hidden_states.shape[0], self.batch_size)

    # =====================================================================
    # forward() Tests — PyramidOutput
    # =====================================================================

    def test_forward_returns_pyramid_output(self):
        """Verify forward returns PyramidOutput dataclass (batch=B)."""
        texts = [f"Solve: {i} + {i+1} = ? Let's think." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)
        self.assertIsInstance(output, PyramidOutput)
        self.assertEqual(output.concepts[0].shape[0], self.batch_size)

    def test_forward_concepts_count(self):
        """Verify PyramidOutput contains all K levels (batch=B)."""
        texts = [f"What is {i} times {i+2}?" for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)
        self.assertEqual(len(output.concepts), self.nlcp_config.num_levels)

    def test_forward_concept_shapes(self):
        """Verify each level's concept shape [B, L_k, D] for any B>=1."""
        texts = [f"Calculate {i} divided by {i+1}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        for level_idx, concepts in enumerate(output.concepts):
            expected_len = self.nlcp_config.level_lengths[level_idx]
            self.assertEqual(
                concepts.shape,
                (self.batch_size, expected_len, self.nlcp_config.hidden_dim),
                f"Level {level_idx} concept shape mismatch",
            )

    def test_forward_level_output_fields(self):
        """Verify each LevelOutput has correct field shapes (batch=B)."""
        texts = [f"Find the sum of {i} and {i+3}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)
        seq_len = enc_out.hidden_states.shape[1]

        for level_idx, level_out in enumerate(output.level_outputs):
            L_k = self.nlcp_config.level_lengths[level_idx]
            with self.subTest(level=level_idx):
                self.assertEqual(
                    level_out.concepts.shape,
                    (self.batch_size, L_k, self.nlcp_config.hidden_dim),
                )
                self.assertEqual(
                    level_out.base_concepts.shape,
                    (self.batch_size, L_k, self.nlcp_config.hidden_dim),
                )
                self.assertEqual(
                    level_out.attention_weights.shape, (self.batch_size, L_k, seq_len)
                )
                self.assertEqual(
                    level_out.reconstruction.shape,
                    (self.batch_size, seq_len, self.nlcp_config.hidden_dim),
                )

    def test_forward_residual_property(self):
        """Verify f_hat + f_rest = H_proj for any B>=1."""
        texts = [f"Compute {i} * {i+1} + {i}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        recomposed = output.reconstructed_hidden + output.residual_hidden
        diff = torch.abs(recomposed - output.projected_hidden).max()
        self.assertLess(
            diff.item(),
            1e-4,
            "f_hat + f_rest should equal H_proj",
        )

    def test_forward_pyramid_properties(self):
        """Verify PyramidOutput metadata and convenience properties (batch=B)."""
        texts = [f"What is {i} minus {i-1}?" for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        # Metadata
        self.assertEqual(output.num_levels, self.nlcp_config.num_levels)
        self.assertEqual(output.level_lengths, self.nlcp_config.level_lengths)

        # total_concepts property
        expected_total = sum(self.nlcp_config.level_lengths)
        self.assertEqual(output.total_concepts, expected_total)

        # all_attentions property
        self.assertEqual(len(output.all_attentions), self.nlcp_config.num_levels)

        # all_base_concepts property
        self.assertEqual(len(output.all_base_concepts), self.nlcp_config.num_levels)

        # cat_concepts
        cat = output.cat_concepts()
        self.assertEqual(
            cat.shape,
            (self.batch_size, expected_total, self.nlcp_config.hidden_dim),
        )

    def test_forward_level0_no_refinement(self):
        """Verify level 0 concepts equal base concepts (batch=B)."""
        texts = [f"Solve: {i} + {i} = ?" for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=16
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        level0 = output.level_outputs[0]
        self.assertTrue(
            torch.allclose(level0.concepts, level0.base_concepts),
            "Level 0 should have no refinement",
        )

    def test_attention_weights_sum_to_one(self):
        """Verify softmax attention sums to 1 per concept slot (batch=B)."""
        texts = [f"Calculate {i} * {i+2}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=16
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        for level_idx, level_out in enumerate(output.level_outputs):
            attn = level_out.attention_weights
            attn_sum = attn.sum(dim=-1)
            self.assertTrue(
                torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5),
                f"Level {level_idx} attention should sum to 1",
            )

    # =====================================================================
    # forward_next_level() Tests
    # =====================================================================

    def test_forward_next_level_returns_single_level_output(self):
        """Verify forward_next_level returns SingleLevelOutput (batch=B)."""
        texts = [f"What is {i} + {i+1}?" for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=16
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        self.builder.clear_cache()

        output = self.builder.forward_next_level(
            enc_out.hidden_states,
            previous_level_concepts=None,
            target_level_index=0,
        )
        self.assertIsInstance(output, SingleLevelOutput)
        self.assertEqual(output.level_index, 0)
        self.assertEqual(output.concepts.shape[0], self.batch_size)

    def test_forward_next_level_sequential_extraction(self):
        """Verify level-by-level extraction on batch=B."""
        texts = [f"Find the product of {i} and {i+2}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=24
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        H = enc_out.hidden_states

        self.builder.clear_cache()
        prev_concepts = []
        for k in range(self.nlcp_config.num_levels):
            level_out = self.builder.forward_next_level(
                H, previous_level_concepts=prev_concepts, target_level_index=k
            )
            prev_concepts.append(level_out.concepts)
            self.assertIsInstance(level_out, SingleLevelOutput)
            self.assertEqual(level_out.level_index, k)
            self.assertEqual(level_out.concepts.shape[0], self.batch_size)
            self.assertEqual(
                level_out.concepts.shape[1],
                self.nlcp_config.level_lengths[k],
            )

        self.assertEqual(
            len(self.builder._cached_attentions),
            self.nlcp_config.num_levels,
        )

    # =====================================================================
    # Cache Tests
    # =====================================================================

    def test_clear_cache(self):
        """Verify clear_cache resets internal state (batch=B)."""
        texts = [f"What is {i} + {i}?" for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=16
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        self.builder.clear_cache()
        self.builder.forward_next_level(
            enc_out.hidden_states,
            previous_level_concepts=None,
            target_level_index=0,
        )
        self.assertEqual(len(self.builder._cached_attentions), 1)

        self.builder.clear_cache()
        self.assertEqual(len(self.builder._cached_attentions), 0)
        self.assertEqual(len(self.builder._cached_base_concepts), 0)

    # =====================================================================
    # GSM8K Integration Tests
    # =====================================================================

    def test_gsm8k_sample_encode_cot(self):
        """Verify encode_cot works on real GSM8K CoT (batch=B)."""
        n = min(self.batch_size, len(self.dataset))
        samples = [self.dataset[i] for i in range(n)]
        cot_texts = [s.cot_answer for s in samples]

        tokens = self.tokenizer(
            cot_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.nlcp_config.max_seq_len,
        )
        input_ids = tokens["input_ids"].to(self.device)

        output = self.builder.encode_cot(input_ids)
        self.assertIsInstance(output, EncoderOutput)
        self.assertEqual(output.hidden_states.shape[0], n)
        self.assertEqual(
            output.hidden_states.shape[-1],
            self.builder.encoder_hidden_dim,
        )

    def test_gsm8k_sample_build_pyramid(self):
        """Verify full pipeline: GSM8K → tokenize → encode → pyramid (batch=B)."""
        n = min(self.batch_size, len(self.dataset))
        samples = [self.dataset[i] for i in range(n)]
        cot_texts = [s.cot_answer for s in samples]

        tokens = self.tokenizer(
            cot_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.nlcp_config.max_seq_len,
        )
        input_ids = tokens["input_ids"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids)
        pyramid = self.builder(enc_out.hidden_states)

        self.assertIsInstance(pyramid, PyramidOutput)
        self.assertEqual(len(pyramid.concepts), self.nlcp_config.num_levels)
        self.assertEqual(pyramid.total_concepts, sum(self.nlcp_config.level_lengths))
        self.assertEqual(pyramid.concepts[0].shape[0], n)

    def test_gsm8k_batch_processing(self):
        """Verify batch processing of multiple GSM8K samples (batch=B)."""
        n = min(self.batch_size, len(self.dataset))
        cot_texts = [self.dataset[i].cot_answer for i in range(n)]

        tokens = self.tokenizer(
            cot_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.nlcp_config.max_seq_len,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        enc_out = self.builder.encode_cot(input_ids, attention_mask)
        pyramid = self.builder(enc_out.hidden_states)

        self.assertEqual(pyramid.concepts[0].shape[0], n)
        self.assertEqual(pyramid.projected_hidden.shape[0], n)

    # =====================================================================
    # Gradient Flow Test
    # =====================================================================

    def test_gradient_flow(self):
        """Verify gradients flow through forward pass on batch=B."""
        texts = [f"Compute {i} + {i+1}." for i in range(self.batch_size)]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=16
        )
        input_ids = tokens["input_ids"].to(self.device)

        self.builder.train()
        enc_out = self.builder.encode_cot(input_ids)
        output = self.builder(enc_out.hidden_states)

        loss = sum(c.sum() for c in output.concepts)
        loss.backward()

        self.assertIsNotNone(self.builder.input_proj.weight.grad)
        self.assertIsNotNone(self.builder.temperature.grad)
        for queries in self.builder.concept_queries:
            self.assertIsNotNone(queries.grad)

        self.builder.eval()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    unittest.main(verbosity=2)
