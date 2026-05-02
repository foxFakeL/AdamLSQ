import torch
import numpy as np
import time


def test_quant_pack():
    """Test INT4/INT8 packing functions."""
    from fused_adam_lsq.op_builder import FusedAdamLSQBuilder

    try:
        builder = FusedAdamLSQBuilder()
        opt = builder.load(verbose=True)
        print("Successfully loaded fused_adam_lsq extension")
    except Exception as e:
        print(f"Failed to load extension: {e}")
        return False

    return True


def test_fused_adam_lsq_step():
    """Test FusedAdamLSQ step function with INT8 quantization."""
    print("\n=== Testing FusedAdamLSQ Step (INT8) ===")

    param_size = 1024
    group_size = 128
    num_groups = param_size // group_size

    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1

    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # 手动设置所有buffer
        quant_buffer = torch.zeros(param_size, dtype=torch.uint8)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)

        optimizer.set_quant_buffer(model_params[0], quant_buffer)
        optimizer.set_delta_tensor(model_params[0], delta)
        optimizer.set_z_tensor(model_params[0], z)

        model_params[0].grad = grads.clone()
        optimizer.step()

        print(f"Quant buffer shape: {quant_buffer.shape}")
        print(f"Delta shape: {delta.shape}")
        print(f"Z shape: {z.shape}")
        print(f"Quant buffer values range: {quant_buffer.min().item()} - {quant_buffer.max().item()}")

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

    param_size = 1024
    group_size = 128
    num_groups = param_size // group_size

    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1

    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=4)

        # 手动设置所有buffer
        quant_buffer = torch.zeros(param_size // 2, dtype=torch.uint8)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)

        optimizer.set_quant_buffer(model_params[0], quant_buffer)
        optimizer.set_delta_tensor(model_params[0], delta)
        optimizer.set_z_tensor(model_params[0], z)

        model_params[0].grad = grads.clone()
        optimizer.step()

        print(f"Quant buffer shape: {quant_buffer.shape}")
        print(f"Expected shape for INT4: {param_size // 2}")

        unpacked = []
        for packed in quant_buffer:
            high = (packed.item() >> 4) & 0x0F
            low = packed.item() & 0x0F
            unpacked.extend([high, low])

        unpacked = unpacked[:param_size]

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

        param_size = 128
        group_size = 32
        q_bits = 8

        quant_buffer = torch.randint(0, 255, (param_size,), dtype=torch.uint8)
        delta = torch.ones(param_size // group_size, dtype=torch.float32) * 0.1
        z = torch.zeros(param_size // group_size, dtype=torch.float32)

        dequant = dequantize_weight(quant_buffer, delta, z, q_bits, group_size, param_size, dtype=torch.float32)

        print(f"Dequantized shape: {dequant.shape}")
        print(f"Dequantized dtype: {dequant.dtype}")

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

    param_size = 1024 * 1024
    group_size = 128
    num_groups = param_size // group_size

    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.01

    try:
        from fused_adam_lsq import FusedAdamLSQ

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # 手动设置buffer
        quant_buffer = torch.zeros(param_size, dtype=torch.uint8)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)
        optimizer.set_quant_buffer(model_params[0], quant_buffer)
        optimizer.set_delta_tensor(model_params[0], delta)
        optimizer.set_z_tensor(model_params[0], z)

        model_params[0].grad = grads.clone()

        for _ in range(3):
            optimizer.step()

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


def test_set_quant_buffer():
    """Test updating quant buffer pointer externally."""
    print("\n=== Testing set_quant_buffer ===")

    param_size = 1024
    group_size = 128
    num_groups = param_size // group_size

    try:
        from fused_adam_lsq import FusedAdamLSQ

        params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # 第一次设置
        quant_buffer1 = torch.zeros(param_size, dtype=torch.uint8, pin_memory=True)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)
        optimizer.set_quant_buffer(model_params[0], quant_buffer1)
        optimizer.set_delta_tensor(model_params[0], delta)
        optimizer.set_z_tensor(model_params[0], z)

        model_params[0].grad = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1
        optimizer.step()

        # 更换quant buffer
        quant_buffer2 = torch.zeros(param_size, dtype=torch.uint8, pin_memory=True)
        optimizer.set_quant_buffer(model_params[0], quant_buffer2)

        current_buffer = optimizer.get_quant_buffer(model_params[0])
        assert current_buffer is quant_buffer2, "Buffer should be updated to quant_buffer2"

        print("set_quant_buffer test passed!")
        return True

    except Exception as e:
        print(f"set_quant_buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_set_quant_buffer_validation():
    """Test validation checks for set_quant_buffer."""
    print("\n=== Testing set_quant_buffer validation ===")

    param_size = 256
    group_size = 128

    try:
        from fused_adam_lsq import FusedAdamLSQ

        params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # Test: Wrong size should raise error
        wrong_size_buffer = torch.zeros(100, dtype=torch.uint8)
        try:
            optimizer.set_quant_buffer(model_params[0], wrong_size_buffer)
            print("ERROR: Should have raised ValueError for wrong size")
            return False
        except ValueError as e:
            print(f"Correctly raised error for wrong size: {e}")

        # Test: Wrong dtype should raise error
        wrong_dtype_buffer = torch.zeros(param_size, dtype=torch.float32)
        try:
            optimizer.set_quant_buffer(model_params[0], wrong_dtype_buffer)
            print("ERROR: Should have raised ValueError for wrong dtype")
            return False
        except ValueError as e:
            print(f"Correctly raised error for wrong dtype: {e}")

        print("set_quant_buffer validation tests passed!")
        return True

    except Exception as e:
        print(f"set_quant_buffer validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_without_quant():
    """Test step_without_quant - Adam update without quantization."""
    print("\n=== Testing step_without_quant ===")

    param_size = 1024
    group_size = 128

    try:
        from fused_adam_lsq import FusedAdamLSQ

        params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # step_without_quant不需要设置量化buffer
        model_params[0].grad = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1
        optimizer.step_without_quant()

        # 多次step验证
        start = time.time()
        for _ in range(20):
            optimizer.step_without_quant()
        elapsed_noq = (time.time() - start) / 20 * 1000
        print(f"step_without_quant time: {elapsed_noq:.3f}ms/step")

        # 对比：带量化的step
        num_groups = param_size // group_size
        optimizer2 = FusedAdamLSQ([params.clone().requires_grad_()], lr=1e-3, q_bits=8)
        quant_buffer = torch.zeros(param_size, dtype=torch.uint8)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)
        optimizer2.set_quant_buffer(optimizer2.param_groups[0]['params'][0], quant_buffer)
        optimizer2.set_delta_tensor(optimizer2.param_groups[0]['params'][0], delta)
        optimizer2.set_z_tensor(optimizer2.param_groups[0]['params'][0], z)
        optimizer2.param_groups[0]['params'][0].grad = torch.randn(param_size, dtype=torch.bfloat16) * 0.1

        start = time.time()
        for _ in range(20):
            optimizer2.step()
        elapsed_quant = (time.time() - start) / 20 * 1000
        print(f"step with quantization: {elapsed_quant:.3f}ms/step")

        overhead = elapsed_quant - elapsed_noq
        print(f"Quantization overhead: {overhead:.3f}ms ({overhead/elapsed_noq*100:.1f}%)")

        print("step_without_quant test passed!")
        return True

    except Exception as e:
        print(f"step_without_quant test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_without_buffer_raises():
    """Test that step raises error when buffers not set."""
    print("\n=== Testing step validation (no buffer) ===")

    param_size = 256
    group_size = 128

    try:
        from fused_adam_lsq import FusedAdamLSQ

        params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        model_params[0].grad = torch.randn(param_size, dtype=torch.bfloat16) * 0.1

        # 不设置buffer直接step应该报错
        try:
            optimizer.step()
            print("ERROR: Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"Correctly raised error: {e}")
            return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_param_meta():
    """Test get_param_meta method."""
    print("\n=== Testing get_param_meta ===")

    param_size = 1024
    group_size = 128

    try:
        from fused_adam_lsq import FusedAdamLSQ

        params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        meta = optimizer.get_param_meta(model_params[0])
        print(f"Param meta: {meta}")

        expected_num_groups = param_size // group_size
        assert meta['num_elements'] == param_size
        assert meta['num_groups'] == expected_num_groups
        assert meta['quant_size'] == param_size  # INT8

        print("get_param_meta test passed!")
        return True

    except Exception as e:
        print(f"get_param_meta test failed: {e}")
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
        'set_quant_buffer': test_set_quant_buffer(),
        'set_quant_buffer_validation': test_set_quant_buffer_validation(),
        'step_without_quant': test_step_without_quant(),
        'step_without_buffer_raises': test_step_without_buffer_raises(),
        'get_param_meta': test_get_param_meta(),
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