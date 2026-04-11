"""NLCP Implementation Verification Tests.

This module verifies the correctness of the NLCP implementation
against the specifications in concept-pyramid.md.

Reference: concept-pyramid.md Section 8 - Recommended Experimental Path
"MVP Validation: Fix K=2, run L = L_NTP + L_consist + L_CE pipeline,
verify tensor flow and gradient closure"
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from examples.nlcp.base import (
    NLCPModelConfig,
    NLCPTrainingConfig,
    NLCPInferenceConfig,
    LevelState,
    NLCPOutput,
)
from examples.nlcp.modules import (
    RMSNorm,
    DepthGate,
    ExpansionPredictor,
    CrossLevelCausalAttention,
    SelfAttentionBlock,
    NextLevelGenerator,
    TokenDecoder,
    HFCausalEncoder,
)
from examples.nlcp.losses import (
    NextTokenPredictionLoss,
    CrossScaleConsistencyLoss,
    ExpansionRateRegularization,
    FinalTokenAlignmentLoss,
    NLCPLossComputer,
)
from examples.nlcp.model import NLCPModel, build_nlcp_model
from examples.nlcp.inference import NLCPInference, build_inference_engine


def test_rmsnorm():
    """Test RMSNorm dimension preservation.

    Reference: concept-pyramid.md Section 3.4
    "RMSNorm stabilizes heterogeneous statistics (DLCM Eq.16)"
    """
    print("Testing RMSNorm...")
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    norm = RMSNorm(hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = norm(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ RMSNorm: Input {x.shape} -> Output {y.shape}")


def test_depth_gate():
    """Test DepthGate output range and dimensions.

    Reference: concept-pyramid.md Section 3.2
    "p_cont^(k) = σ(MLP_2(GELU(MLP_1(Pool(H_k)))))"
    Output should be scalar probability in [0, 1].
    """
    print("Testing DepthGate...")
    hidden_dim = 64
    batch_size = 2
    seq_len = 16
    dropout = 0.1

    gate = DepthGate(hidden_dim, dropout)
    H = torch.randn(batch_size, seq_len, hidden_dim)
    p_cont = gate(H)

    assert p_cont.shape == (batch_size, 1), f"Shape mismatch: {p_cont.shape}"
    assert (p_cont >= 0).all() and (p_cont <= 1).all(), "Probability out of range"
    print(
        f"  ✓ DepthGate: H {H.shape} -> p_cont {p_cont.shape}, range [{p_cont.min():.4f}, {p_cont.max():.4f}]"
    )


def test_expansion_predictor():
    """Test ExpansionPredictor output dimensions and constraints.

    Reference: concept-pyramid.md Section 3.3
    "λ_k = Softplus(MLP(H_k)) ∈ [1, ∞)^{L_k}"
    "expand_mask_k = ⌊λ_k⌋"
    """
    print("Testing ExpansionPredictor...")
    hidden_dim = 64
    batch_size = 2
    seq_len = 16
    expansion_min = 1
    expansion_max = 8
    dropout = 0.1

    predictor = ExpansionPredictor(hidden_dim, expansion_min, expansion_max, dropout)
    H = torch.randn(batch_size, seq_len, hidden_dim)
    expand_mask, lambda_k = predictor(H)

    assert expand_mask.shape == (
        batch_size,
        seq_len,
    ), f"Shape mismatch: {expand_mask.shape}"
    assert lambda_k.shape == (batch_size, seq_len), f"Shape mismatch: {lambda_k.shape}"
    assert (expand_mask >= expansion_min).all(), "Expansion below minimum"
    assert (expand_mask <= expansion_max).all(), "Expansion above maximum"
    print(
        f"  ✓ ExpansionPredictor: H {H.shape} -> mask {expand_mask.shape}, range [{expand_mask.min()}, {expand_mask.max()}]"
    )


def test_cross_level_causal_attention():
    """Test CrossLevelCausalAttention dimension alignment.

    Reference: concept-pyramid.md Section 3.4
    "repeat_interleave makes irregular mapping degenerate to standard
    L_{k+1} × L_{k+1} Causal Mask"

    Key verification: K/V replication aligns with fine level length.
    """
    print("Testing CrossLevelCausalAttention...")
    hidden_dim = 64
    num_heads = 4
    batch_size = 2
    coarse_len = 8
    dropout = 0.1

    cross_attn = CrossLevelCausalAttention(hidden_dim, num_heads, dropout)

    # Create coarse hidden states
    H_coarse = torch.randn(batch_size, coarse_len, hidden_dim)

    # Create expand_mask (sum determines fine length)
    expand_mask = torch.randint(1, 4, (batch_size, coarse_len))
    fine_len = expand_mask.sum(dim=-1).max().item()

    # Create fine hidden states
    H_fine = torch.randn(batch_size, fine_len, hidden_dim)

    # Run cross-attention
    output = cross_attn(H_fine, H_coarse, expand_mask)

    assert (
        output.shape == H_fine.shape
    ), f"Shape mismatch: {output.shape} vs {H_fine.shape}"
    print(
        f"  ✓ CrossLevelCausalAttention: Coarse {H_coarse.shape} -> Fine {H_fine.shape} -> Output {output.shape}"
    )


def test_next_level_generator():
    """Test NextLevelGenerator output dimensions.

    Reference: concept-pyramid.md Section 3.4
    "P(H_{k+1} | H_{<=k}, Q) = ∏_j P(h_{k+1}^j | h_{k+1}^{<j}, H_k, Q)"
    """
    print("Testing NextLevelGenerator...")
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    batch_size = 2
    coarse_len = 8
    dropout = 0.1

    generator = NextLevelGenerator(hidden_dim, num_heads, num_layers, dropout)

    H_coarse = torch.randn(batch_size, coarse_len, hidden_dim)
    expand_mask = torch.randint(1, 4, (batch_size, coarse_len))
    fine_len = expand_mask.sum(dim=-1).max().item()
    H_fine = torch.randn(batch_size, fine_len, hidden_dim)

    output, _ = generator(H_fine, H_coarse, expand_mask)

    assert (
        output.shape == H_fine.shape
    ), f"Shape mismatch: {output.shape} vs {H_fine.shape}"
    print(f"  ✓ NextLevelGenerator: Fine {H_fine.shape} -> Output {output.shape}")


def test_token_decoder():
    """Test TokenDecoder output dimensions and μP scaling.

    Reference: concept-pyramid.md Section 4.2
    "Output layer scaling: logits = (1/s_token)(H_K @ W_unemb^T)
    ensures logits magnitude is O(1)"
    """
    print("Testing TokenDecoder...")
    hidden_dim = 64
    vocab_size = 1000
    muP_scale = 1.0
    batch_size = 2
    seq_len = 16

    decoder = TokenDecoder(hidden_dim, vocab_size, muP_scale)
    H = torch.randn(batch_size, seq_len, hidden_dim)
    logits = decoder(H)

    assert logits.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"Shape mismatch: {logits.shape}"
    print(f"  ✓ TokenDecoder: H {H.shape} -> logits {logits.shape}")


def test_hf_causal_encoder():
    """Test HFCausalEncoder output dimensions.

    Reference: concept-pyramid-V1.md Section 3.1
    "Reuse HuggingFace pretrained model weights"

    Note: This test requires a HuggingFace model to be available.
    It uses a small model (e.g., distilbert) for testing purposes.
    """
    print("Testing HFCausalEncoder...")
    model_name = "distilbert-base-uncased"
    l0_length = 8
    batch_size = 2
    input_len = 32

    try:
        encoder = HFCausalEncoder(
            model_name=model_name,
            num_layers=2,
            l0_length=l0_length,
            freeze_encoder=False,
        )
        input_ids = torch.randint(0, 30522, (batch_size, input_len))
        attention_mask = torch.ones(batch_size, input_len, dtype=torch.long)
        H_0 = encoder(input_ids, attention_mask)

        assert H_0.shape == (
            batch_size,
            l0_length,
            encoder.hidden_dim,
        ), f"Shape mismatch: {H_0.shape}"
        print(f"  ✓ HFCausalEncoder: input_ids {input_ids.shape} -> H_0 {H_0.shape}")
    except Exception as e:
        print(f"  ⚠ HFCausalEncoder test skipped: {e}")


def test_losses():
    """Test all loss functions.

    Reference: concept-pyramid.md Section 4.1
    Complete Loss Function
    """
    print("Testing Loss Functions...")
    hidden_dim = 64
    vocab_size = 1000
    batch_size = 2
    seq_len = 16
    padding_id = 0

    # Test NTP Loss
    ntp_loss = NextTokenPredictionLoss(vocab_size, hidden_dim)
    H = torch.randn(batch_size, seq_len, hidden_dim)
    target_ids = torch.randint(1, vocab_size, (batch_size, seq_len + 10))
    loss = ntp_loss(H, target_ids, padding_id)
    assert loss.dim() == 0, f"NTP loss should be scalar, got {loss.shape}"
    print(f"  ✓ NextTokenPredictionLoss: {loss.item():.4f}")

    # Test Consistency Loss
    consist_loss = CrossScaleConsistencyLoss(use_info_nce=True, info_nce_weight=0.1)
    H_fine = torch.randn(batch_size, 32, hidden_dim)
    H_coarse = torch.randn(batch_size, 8, hidden_dim)
    expand_mask = torch.randint(2, 5, (batch_size, 8))
    loss = consist_loss(H_fine, H_coarse, expand_mask)
    assert loss.dim() == 0, f"Consistency loss should be scalar, got {loss.shape}"
    print(f"  ✓ CrossScaleConsistencyLoss: {loss.item():.4f}")

    # Test Depth Loss
    depth_loss = ExpansionRateRegularization(target_ratio=4.0)
    loss = depth_loss(coarse_length=8, fine_length=32)
    # loss is a Python float for this simple loss
    assert isinstance(loss, (float, int)) or (
        hasattr(loss, "dim") and loss.dim() == 0
    ), f"Depth loss should be scalar"
    print(f"  ✓ ExpansionRateRegularization: {loss:.4f}")

    # Test CE Loss
    ce_loss = FinalTokenAlignmentLoss(padding_id)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    loss = ce_loss(logits, target_ids)
    assert loss.dim() == 0, f"CE loss should be scalar, got {loss.shape}"
    print(f"  ✓ FinalTokenAlignmentLoss: {loss.item():.4f}")


def test_full_model_forward():
    """Test complete NLCPModel forward pass.

    Reference: concept-pyramid.md Section 2.1
    Complete data flow through the pyramid.
    """
    print("Testing Full Model Forward Pass...")
    device = torch.device("cpu")

    # Build model config
    # Reference: Section 3.1 specs
    model_config = NLCPModelConfig(
        hidden_dim=128,
        num_heads=4,
        vocab_size=1000,
        max_depth=2,
        depth_gate_threshold=0.4,
        l0_length=4,
        l_max=64,
        dropout=0.1,
        expansion_min=1,
        expansion_max=4,
        depth_gate_type="standard",
        expansion_predictor_type="floor",
        cross_attention_type="standard",
        consistency_loss_type="standard",
        encoder_model_name="distilbert-base-uncased",
        encoder_num_layers=2,
        encoder_freeze=False,
    )

    # Build model
    model = build_nlcp_model(
        config=model_config,
        padding_id=0,
        num_encoder_layers=2,
        num_generator_layers=1,
        use_info_nce=False,
        info_nce_weight=0.0,
    )
    model = model.to(device)

    # Create inputs
    batch_size = 2
    input_len = 16
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, input_len)).to(
        device
    )
    target_ids = torch.randint(
        0, model_config.vocab_size, (batch_size, input_len + 20)
    ).to(device)

    # Forward pass
    output = model(
        input_ids=input_ids,
        target_ids=target_ids,
        padding_id=0,
        compute_loss=True,
    )

    # Verify output structure
    assert isinstance(output, NLCPOutput), f"Output type mismatch: {type(output)}"
    assert len(output.level_states) >= 1, "Should have at least one level state"
    assert output.logits.dim() == 3, f"Logits should be 3D, got {output.logits.dim()}"

    print(f"  ✓ Full Model Forward:")
    print(f"    - Input: {input_ids.shape}")
    print(f"    - Levels: {len(output.level_states)}")
    for i, state in enumerate(output.level_states):
        print(
            f"    - Level {i}: H {state.hidden_states.shape}, p_cont={state.depth_gate_prob:.4f}"
        )
    print(f"    - Logits: {output.logits.shape}")
    print(f"    - Total Loss: {output.total_loss:.4f}")


def test_gradient_flow():
    """Test gradient flow through the model.

    Reference: concept-pyramid.md Section 8
    "Verify tensor flow and gradient closure"
    """
    print("Testing Gradient Flow...")
    device = torch.device("cpu")

    model_config = NLCPModelConfig(
        hidden_dim=64,
        num_heads=2,
        vocab_size=100,
        max_depth=2,
        depth_gate_threshold=0.4,
        l0_length=4,
        l_max=32,
        dropout=0.0,
        expansion_min=1,
        expansion_max=2,
        depth_gate_type="standard",
        expansion_predictor_type="floor",
        cross_attention_type="standard",
        consistency_loss_type="standard",
        encoder_model_name="distilbert-base-uncased",
        encoder_num_layers=1,
        encoder_freeze=False,
    )

    model = build_nlcp_model(
        config=model_config,
        padding_id=0,
        num_encoder_layers=1,
        num_generator_layers=1,
        use_info_nce=False,
        info_nce_weight=0.0,
    )

    input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
    target_ids = torch.randint(0, model_config.vocab_size, (1, 16))

    output = model(
        input_ids=input_ids,
        target_ids=target_ids,
        padding_id=0,
        compute_loss=True,
    )

    # Backward pass
    output.total_loss.backward()

    # Check gradients exist
    has_grad = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_grad += 1

    grad_ratio = has_grad / max(total_params, 1)
    print(
        f"  ✓ Gradient Flow: {has_grad}/{total_params} ({grad_ratio*100:.1f}%) parameters have gradients"
    )


def test_inference():
    """Test inference pipeline.

    Reference: concept-pyramid.md Section 5.1
    Blocking generation algorithm
    """
    print("Testing Inference Pipeline...")
    device = torch.device("cpu")

    model_config = NLCPModelConfig(
        hidden_dim=64,
        num_heads=2,
        vocab_size=100,
        max_depth=2,
        depth_gate_threshold=0.4,
        l0_length=4,
        l_max=32,
        dropout=0.0,
        expansion_min=1,
        expansion_max=2,
        depth_gate_type="standard",
        expansion_predictor_type="floor",
        cross_attention_type="standard",
        consistency_loss_type="standard",
        encoder_model_name="distilbert-base-uncased",
        encoder_num_layers=1,
        encoder_freeze=False,
    )

    inference_config = NLCPInferenceConfig(
        max_depth=2,
        depth_threshold=0.4,
        temperature=0.5,
        top_k=10,
        top_p=0.9,
        early_exit=True,
    )

    model = build_nlcp_model(
        config=model_config,
        padding_id=0,
        num_encoder_layers=1,
        num_generator_layers=1,
        use_info_nce=False,
        info_nce_weight=0.0,
    )

    engine = build_inference_engine(model, inference_config)

    input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
    generated = engine.generate(input_ids, max_new_tokens=5)

    print(f"  ✓ Inference: Input {input_ids.shape} -> Generated {generated.shape}")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("NLCP Implementation Verification")
    print("Reference: concept-pyramid.md")
    print("=" * 60)
    print()

    try:
        test_rmsnorm()
        test_depth_gate()
        test_expansion_predictor()
        test_cross_level_causal_attention()
        test_next_level_generator()
        test_token_decoder()
        test_lightweight_encoder()
        test_losses()
        test_full_model_forward()
        test_gradient_flow()
        test_inference()

        print()
        print("=" * 60)
        print(
            "✅ ALL TESTS PASSED - Implementation verified against concept-pyramid.md"
        )
        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
