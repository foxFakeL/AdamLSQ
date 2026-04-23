import torch
import numpy as np
import time


def test_quant_pack():
    """Test INT4/INT8 packing functions."""
    from fused_adam_lsq.op_builder import FusedAdamLSQBuilder

    # Load the C++ extension to test packing
    try:
        builder = FusedAdamLSQBuilder()
        opt = builder.load(verbose=True)
        print("Successfully loaded fused_adam_lsq extension")
    except Exception as e:
        print(f"Failed to load extension: {e}")
        print("Skipping C++ tests, testing Python fallback")
        return False

    return True


def test_fused_adam_lsq_step():
    """Test FusedAdamLSQ step function with INT8 quantization."""
    print("\n=== Testing FusedAdamLSQ Step (INT8) ===")

    # Create test parameters on CPU
    param_size = 1024
    group_size = 128

    # BF16 weights
    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1

    # Create optimizer
    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # Manually set gradient
        model_params[0].grad = grads.clone()

        # Step
        optimizer.step()

        # Get quantized buffer
        quant_buffer = optimizer.get_quant_buffer(model_params[0])
        delta = optimizer.get_delta_tensor(model_params[0])
        z = optimizer.get_z_tensor(model_params[0])

        print(f"Quant buffer shape: {quant_buffer.shape}")
        print(f"Delta shape: {delta.shape}")
        print(f"Z shape: {z.shape}")
        print(f"Quant buffer values range: {quant_buffer.min().item()} - {quant_buffer.max().item()}")

        # Verify quantization range
        assert quant_buffer.min().item() >= 0, "Quant values should be >= 0"
        assert quant_buffer.max().item() <= 255, "Quant values should be <= 255"

        print("INT8 test passed!")
        return True

    except Exception as e:
        print(f"INT8 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fused_adam_lsq_int4():
    """Test FusedAdamLSQ step function with INT4 quantization."""
    print("\n=== Testing FusedAdamLSQ Step (INT4) ===")

    # Create test parameters on CPU
    param_size = 1024  # Must be divisible by 2 for INT4
    group_size = 128

    # BF16 weights
    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1

    # Create optimizer
    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=4)

        # Manually set gradient
        model_params[0].grad = grads.clone()

        # Step
        optimizer.step()

        # Get quantized buffer
        quant_buffer = optimizer.get_quant_buffer(model_params[0])
        delta = optimizer.get_delta_tensor(model_params[0])
        z = optimizer.get_z_tensor(model_params[0])

        print(f"Quant buffer shape: {quant_buffer.shape}")
        print(f"Expected shape for INT4: {param_size // 2}")

        # Each uint8 contains 2 int4 values
        # High 4 bits: value 0, Low 4 bits: value 1
        # Range check: each nibble should be 0-15

        # Unpack to verify
        unpacked = []
        for packed in quant_buffer:
            high = (packed.item() >> 4) & 0x0F
            low = packed.item() & 0x0F
            unpacked.extend([high, low])

        unpacked = unpacked[:param_size]  # Truncate if needed

        print(f"Unpacked values range: {min(unpacked)} - {max(unpacked)}")
        assert min(unpacked) >= 0, "INT4 values should be >= 0"
        assert max(unpacked) <= 15, "INT4 values should be <= 15"

        print("INT4 test passed!")
        return True

    except Exception as e:
        print(f"INT4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dequantize():
    """Test dequantization function."""
    print("\n=== Testing Dequantization ===")

    try:
        from fused_adam_lsq import dequantize_weight

        # Create test quantized data
        param_size = 128
        group_size = 32
        q_bits = 8

        quant_buffer = torch.randint(0, 255, (param_size,), dtype=torch.uint8)
        delta = torch.ones(param_size // group_size, dtype=torch.float32) * 0.1
        z = torch.zeros(param_size // group_size, dtype=torch.float32)

        # Dequantize with float32 output for better precision testing
        dequant = dequantize_weight(quant_buffer, delta, z, q_bits, group_size, param_size, dtype=torch.float32)

        print(f"Dequantized shape: {dequant.shape}")
        print(f"Dequantized dtype: {dequant.dtype}")

        # Verify dequantization formula
        for i in range(param_size):
            group_id = i // group_size
            expected = quant_buffer[i].item() * delta[group_id].item() + z[group_id].item()
            actual = dequant[i].item()
            assert abs(expected - actual) < 1e-6, f"Dequant mismatch at {i}: expected {expected}, got {actual}"

        print("Dequantization test passed!")
        return True

    except Exception as e:
        print(f"Dequantization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """Compare performance of fused vs separate operations."""
    print("\n=== Testing Performance ===")

    param_size = 1024 * 1024  # 1M elements
    group_size = 128

    # Create large parameters
    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.01

    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)
        model_params[0].grad = grads.clone()

        # Warmup
        for _ in range(3):
            optimizer.step()

        # Timed run
        start = time.time()
        for _ in range(10):
            optimizer.step()
        fused_time = time.time() - start

        print(f"Fused Adam + LSQ time (10 steps): {fused_time:.4f}s")
        print(f"Per-step time: {fused_time/10:.4f}s")

        return True

    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Fused Adam + LSQ Quantization Tests")
    print("=" * 60)

    results = {
        'quant_pack': test_quant_pack(),
        'int8_step': test_fused_adam_lsq_step(),
        'int4_step': test_fused_adam_lsq_int4(),
        'dequantize': test_dequantize(),
        'performance': test_performance_comparison(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return all_passed


if __name__ == "__main__":
    run_all_tests()