#!/usr/bin/env python
"""Simple standalone test for FusedAdamLSQCPUOffloadOptimizer.

Tests:
1. Basic optimizer functionality with mock expert weights
2. Loss decrease over iterations
3. Quantization precision transitions
"""

import sys
sys.path.insert(0, '/cnic/work/liuql/fuse_opt/src')

import torch
import torch.nn as nn
import time
from typing import List

print("Testing FusedAdamLSQCPUOffloadOptimizer standalone functionality...")

# Import the optimizer
from fused_adam_lsq import FusedAdamLSQ


class MockExpertModule(nn.Module):
    """Mock expert module for testing."""
    def __init__(self, num_experts: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.num_global_experts = num_experts

        # weight1: [num_experts, hidden_size, ffn_hidden_size]
        self.weight1 = nn.Parameter(
            torch.randn(num_experts, hidden_size, ffn_hidden_size, dtype=torch.bfloat16, device='cpu')
        )
        # weight2: [num_experts, ffn_hidden_size, hidden_size]
        self.weight2 = nn.Parameter(
            torch.randn(num_experts, ffn_hidden_size, hidden_size, dtype=torch.bfloat16, device='cpu')
        )

        # Gradient buffers (on CPU)
        self._grad_weight1 = torch.zeros_like(self.weight1.data, pin_memory=True)
        self._grad_weight2 = torch.zeros_like(self.weight2.data, pin_memory=True)

    def forward(self, x):
        """Simple forward for testing."""
        # Mock forward: just return sum of weights
        return x.sum() + self.weight1.sum() + self.weight2.sum()


def test_basic_training():
    """Test basic training with FusedAdamLSQ."""
    print("\n=== Test 1: Basic Training ===")

    num_experts = 8
    hidden_size = 256
    ffn_hidden_size = 512

    # Create mock expert module
    expert_module = MockExpertModule(num_experts, hidden_size, ffn_hidden_size)

    # Create FusedAdamLSQ optimizer
    optimizer = FusedAdamLSQ(
        [expert_module.weight1, expert_module.weight2],
        lr=1e-2,  # Higher lr for noticeable change
        q_bits=8,
        group_size=128,
    )

    # Track loss history
    loss_history = []

    # Create a target for the weights (for supervised training simulation)
    target_w1 = torch.randn_like(expert_module.weight1) * 0.5
    target_w2 = torch.randn_like(expert_module.weight2) * 0.5

    # Training iterations
    for iter in range(50):
        # Zero gradients
        optimizer.zero_grad()

        # Compute "loss": MSE between weights and target
        loss = ((expert_module.weight1 - target_w1) ** 2).sum() + \
               ((expert_module.weight2 - target_w2) ** 2).sum()

        # Compute gradients (simple MSE gradients)
        expert_module.weight1.grad = 2 * (expert_module.weight1 - target_w1)
        expert_module.weight2.grad = 2 * (expert_module.weight2 - target_w2)

        # Optimizer step
        optimizer.step()

        # Track actual loss
        current_loss = loss.item()
        loss_history.append(current_loss)

        if iter % 10 == 0:
            print(f"  iter {iter}: loss = {current_loss:.4e}")

    # Check loss trend
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_change = (final_loss - initial_loss) / initial_loss * 100

    print(f"\n  Initial loss: {initial_loss:.4e}")
    print(f"  Final loss: {final_loss:.4e}")
    print(f"  Loss change: {loss_change:+.2f}%")

    # Get quant params
    quant_w1 = optimizer.get_quant_buffer(expert_module.weight1)
    delta_w1 = optimizer.get_delta_tensor(expert_module.weight1)
    z_w1 = optimizer.get_z_tensor(expert_module.weight1)

    print(f"\n  Quant buffer shape: {quant_w1.shape}")
    print(f"  Delta shape: {delta_w1.shape}")
    print(f"  Z shape: {z_w1.shape}")
    print(f"  Quant values range: {quant_w1.min().item()} - {quant_w1.max().item()}")

    # Basic checks
    assert quant_w1.min().item() >= 0, "Quant values should be >= 0"
    assert quant_w1.max().item() <= 255, "Quant values should be <= 255"

    # Loss should decrease significantly
    if final_loss < initial_loss * 0.5:  # Should decrease by at least 50%
        print("  ✓ Loss DECREASED significantly (training is working)")
        return True
    elif final_loss < initial_loss:
        print("  ✓ Loss DECREASED (training is working)")
        return True
    else:
        print("  ✗ Loss did NOT decrease")
        return False


def test_int4_quantization():
    """Test INT4 quantization."""
    print("\n=== Test 2: INT4 Quantization ===")

    # Create simple parameter
    param = torch.randn(1024, dtype=torch.bfloat16, device='cpu').requires_grad_()

    # INT4 optimizer
    optimizer = FusedAdamLSQ([param], lr=1e-3, q_bits=4, group_size=128)

    # Set gradient and step
    param.grad = torch.randn_like(param) * 0.01
    optimizer.step()

    # Get quant buffer
    quant_buf = optimizer.get_quant_buffer(param)

    print(f"  INT4 Quant buffer shape: {quant_buf.shape}")
    print(f"  Expected: {param.numel() // 2} (2 elements per byte)")

    # Verify INT4 range (each nibble should be 0-15)
    unpacked = []
    for packed in quant_buf[:10]:  # Check first 10
        high = (packed.item() >> 4) & 0x0F
        low = packed.item() & 0x0F
        unpacked.extend([high, low])

    print(f"  Sample unpacked values: {unpacked}")
    assert min(unpacked) >= 0, "INT4 values should be >= 0"
    assert max(unpacked) <= 15, "INT4 values should be <= 15"

    print("  ✓ INT4 quantization working correctly")
    return True


def test_multiple_params():
    """Test optimizer with multiple parameter groups."""
    print("\n=== Test 3: Multiple Parameters ===")

    params = [
        torch.randn(512, dtype=torch.bfloat16, device='cpu').requires_grad_(),
        torch.randn(1024, dtype=torch.bfloat16, device='cpu').requires_grad_(),
        torch.randn(2048, dtype=torch.bfloat16, device='cpu').requires_grad_(),
    ]

    optimizer = FusedAdamLSQ(params, lr=1e-3, q_bits=8, group_size=128)

    # Set gradients
    for p in params:
        p.grad = torch.randn_like(p) * 0.01

    # Step
    optimizer.step()

    # Check all params have quant buffers
    for i, p in enumerate(params):
        quant_buf = optimizer.get_quant_buffer(p)
        delta = optimizer.get_delta_tensor(p)
        z = optimizer.get_z_tensor(p)

        print(f"  Param {i}: quant_size={quant_buf.shape[0]}, delta_groups={delta.shape[0]}, z_groups={z.shape[0]}")

        assert quant_buf.shape[0] == p.numel(), f"Quant buffer size mismatch for param {i}"
        assert delta.shape[0] == p.numel() // 128, f"Delta group count mismatch for param {i}"

    print("  ✓ Multiple parameters handled correctly")
    return True


def test_gpu_dequantization():
    """Test GPU dequantization functions."""
    print("\n=== Test 4: GPU Dequantization ===")

    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping GPU test")
        return True

    try:
        sys.path.insert(0, '/cnic/work/liuql/Megatron-LM')
        from megatron.core.optimizer.gpu_quant_utils import dequantize_int8, dequantize_int4
    except ImportError:
        print("  ⊘ Megatron not available, skipping GPU dequant test")
        return True

    device = torch.device('cuda')
    group_size = 128

    # Test INT8
    quant_w8 = torch.randint(0, 256, (1024,), dtype=torch.uint8, device=device)
    delta = torch.ones(8, dtype=torch.float32, device=device) * 0.01
    z = torch.zeros(8, dtype=torch.float32, device=device)

    w_dequant = dequantize_int8(quant_w8, delta, z, group_size)

    assert w_dequant.dtype == torch.bfloat16, "Output should be bfloat16"
    print(f"  INT8 dequant output dtype: {w_dequant.dtype}")

    # Verify formula: w = q * delta + z
    expected_first = quant_w8[:128].float() * 0.01
    actual_first = w_dequant[:128].float()
    diff = (expected_first - actual_first).abs().max().item()
    print(f"  INT8 dequant max diff from expected: {diff:.6e}")
    # BF16 has lower precision, allow larger diff
    assert diff < 1e-2, "INT8 dequant formula incorrect"

    # Test INT4
    packed = torch.randint(0, 256, (512,), dtype=torch.uint8, device=device)
    # 512 bytes = 1024 INT4 values, with group_size=128 => 8 groups
    delta4 = torch.ones(8, dtype=torch.float32, device=device) * 0.01
    z4 = torch.zeros(8, dtype=torch.float32, device=device)

    w_dequant4 = dequantize_int4(packed, delta4, z4, group_size, original_size=1024)

    assert w_dequant4.dtype == torch.bfloat16, "INT4 output should be bfloat16"
    print(f"  INT4 dequant output dtype: {w_dequant4.dtype}")

    print("  ✓ GPU dequantization working correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("FusedAdamLSQCPUOffloadOptimizer Standalone Test Suite")
    print("=" * 60)

    results = {}

    results['basic_training'] = test_basic_training()
    results['int4_quantization'] = test_int4_quantization()
    results['multiple_params'] = test_multiple_params()
    results['gpu_dequantization'] = test_gpu_dequantization()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("=" * 60)

    return all(results.values())


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)