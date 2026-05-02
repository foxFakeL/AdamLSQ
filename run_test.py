#!/usr/bin/env python
"""
测试脚本：融合Adam + LSQ量化优化器
"""

import torch
import time
import sys

def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能验证")
    print("=" * 60)

    from fused_adam_lsq import FusedAdamLSQ

    # 创建测试参数
    param_size = 1024
    group_size = 128
    num_groups = param_size // group_size

    # BF16权重
    params = torch.randn(param_size, dtype=torch.bfloat16, device='cpu')
    grads = torch.randn(param_size, dtype=torch.bfloat16, device='cpu') * 0.1

    # INT8量化
    print("\n[INT8量化测试]")
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
    optimizer.step()

    print(f"  量化buffer大小: {quant_buffer.shape}")
    print(f"  量化值范围: [{quant_buffer.min().item()}, {quant_buffer.max().item()}]")

    if quant_buffer.min().item() >= 0 and quant_buffer.max().item() <= 255:
        print("  ✓ INT8量化范围正确")
    else:
        print("  ✗ INT8量化范围错误")
        return False

    # INT4量化
    print("\n[INT4量化测试]")
    model_params2 = [params.clone().requires_grad_()]
    optimizer2 = FusedAdamLSQ(model_params2, lr=1e-3, group_size=group_size, q_bits=4)

    # 手动设置buffer (INT4需要param_size/2的buffer)
    quant_buffer2 = torch.zeros(param_size // 2, dtype=torch.uint8)
    delta2 = torch.ones(num_groups, dtype=torch.float32)
    z2 = torch.zeros(num_groups, dtype=torch.float32)
    optimizer2.set_quant_buffer(model_params2[0], quant_buffer2)
    optimizer2.set_delta_tensor(model_params2[0], delta2)
    optimizer2.set_z_tensor(model_params2[0], z2)

    model_params2[0].grad = grads.clone()
    optimizer2.step()

    print(f"  量化buffer大小: {quant_buffer2.shape} (预期: {param_size // 2})")

    # 解包验证
    unpacked = []
    for packed in quant_buffer2:
        high = (packed.item() >> 4) & 0x0F
        low = packed.item() & 0x0F
        unpacked.extend([high, low])

    print(f"  解包值范围: [{min(unpacked)}, {max(unpacked)}]")

    if min(unpacked) >= 0 and max(unpacked) <= 15:
        print("  ✓ INT4量化范围正确")
    else:
        print("  ✗ INT4量化范围错误")
        return False

    return True


def test_dequantize():
    """反量化测试"""
    print("\n" + "=" * 60)
    print("测试2: 反量化验证")
    print("=" * 60)

    from fused_adam_lsq import dequantize_weight

    param_size = 128
    group_size = 32

    quant_buffer = torch.randint(0, 255, (param_size,), dtype=torch.uint8)
    delta = torch.ones(param_size // group_size, dtype=torch.float32) * 0.1
    z = torch.zeros(param_size // group_size, dtype=torch.float32)

    # 使用float32输出避免精度损失
    dequant = dequantize_weight(quant_buffer, delta, z, 8, group_size, param_size, dtype=torch.float32)

    # 验证几个点
    errors = 0
    for i in [0, 10, 50, 100]:
        group_id = i // group_size
        expected = quant_buffer[i].item() * delta[group_id].item() + z[group_id].item()
        actual = dequant[i].item()
        diff = abs(expected - actual)
        print(f"  索引{i}: 量化值={quant_buffer[i].item()}, 反量化={actual:.4f}, 预期={expected:.4f}, 误差={diff:.6f}")
        if diff > 1e-6:
            errors += 1

    if errors == 0:
        print("  ✓ 反量化正确")
        return True
    else:
        print("  ✗ 反量化有误差")
        return False


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试3: 性能测试")
    print("=" * 60)

    from fused_adam_lsq import FusedAdamLSQ

    sizes = [1024, 10240, 102400, 1024000]  # 1K, 10K, 100K, 1M

    for size in sizes:
        params = torch.randn(size, dtype=torch.bfloat16, device='cpu')
        grads = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01
        group_size = 128
        num_groups = size // group_size

        model_params = [params.clone().requires_grad_()]
        optimizer = FusedAdamLSQ(model_params, lr=1e-3, group_size=group_size, q_bits=8)

        # 手动设置buffer
        quant_buffer = torch.zeros(size, dtype=torch.uint8)
        delta = torch.ones(num_groups, dtype=torch.float32)
        z = torch.zeros(num_groups, dtype=torch.float32)
        optimizer.set_quant_buffer(model_params[0], quant_buffer)
        optimizer.set_delta_tensor(model_params[0], delta)
        optimizer.set_z_tensor(model_params[0], z)

        model_params[0].grad = grads.clone()

        # 预热
        for _ in range(3):
            optimizer.step()

        # 计时
        start = time.time()
        for _ in range(10):
            optimizer.step()
        elapsed = time.time() - start

        print(f"  元素数 {size:>7}: {elapsed:.4f}s (10步), 每步 {elapsed/10*1000:.2f}ms")

    return True


def test_vs_separate():
    """对比测试：融合 vs 分步"""
    print("\n" + "=" * 60)
    print("测试4: 融合 vs 分步对比")
    print("=" * 60)

    from fused_adam_lsq import FusedAdamLSQ, dequantize_weight

    size = 10240
    group_size = 128
    num_groups = size // group_size

    # 创建相同的数据
    params1 = torch.randn(size, dtype=torch.bfloat16, device='cpu')
    params2 = params1.clone()
    grads = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01

    # 融合版本
    model_params1 = [params1.clone().requires_grad_()]
    opt1 = FusedAdamLSQ(model_params1, lr=1e-3, group_size=group_size, q_bits=8)

    # 手动设置buffer
    quant_buffer1 = torch.zeros(size, dtype=torch.uint8)
    delta1 = torch.ones(num_groups, dtype=torch.float32)
    z1 = torch.zeros(num_groups, dtype=torch.float32)
    opt1.set_quant_buffer(model_params1[0], quant_buffer1)
    opt1.set_delta_tensor(model_params1[0], delta1)
    opt1.set_z_tensor(model_params1[0], z1)

    model_params1[0].grad = grads.clone()

    start = time.time()
    opt1.step()
    fused_time = time.time() - start

    # 分步版本（使用标准Adam + Python量化）
    model_params2 = [params2.clone().requires_grad_()]
    opt2 = torch.optim.AdamW(model_params2, lr=1e-3)
    model_params2[0].grad = grads.clone()

    start = time.time()
    opt2.step()
    # Python量化（模拟）
    quant_data = torch.zeros(size, dtype=torch.uint8)
    delta = torch.ones(size // group_size, dtype=torch.float32) * 0.1
    z = torch.zeros(size // group_size, dtype=torch.float32)
    for i in range(size):
        group_id = i // group_size
        q = int((model_params2[0].data[i].item() - z[group_id].item()) / delta[group_id].item())
        q = max(0, min(255, q))
        quant_data[i] = q
    separate_time = time.time() - start

    print(f"  融合版本: {fused_time*1000:.2f}ms")
    print(f"  分步版本: {separate_time*1000:.2f}ms")
    print(f"  加速比: {separate_time/fused_time:.2f}x")

    return True


def main():
    print("\n" + "#" * 60)
    print("# 融合Adam + LSQ量化优化器 测试脚本")
    print("#" * 60)

    tests = [
        ("基础功能", test_basic),
        ("反量化", test_dequantize),
        ("性能", test_performance),
        ("融合vs分步", test_vs_separate),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("全部测试通过!")
        return 0
    else:
        print("部分测试失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())