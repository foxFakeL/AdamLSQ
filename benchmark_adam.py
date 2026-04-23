#!/usr/bin/env python
"""
性能对比测试：DeepSpeed CPU Adam vs Fused Adam LSQ

测试目标：
1. 纯Adam性能对比
2. 量化开销分析
3. 不同参数规模下的性能差异
"""

import torch
import time
import sys
import os

# 添加DeepSpeed路径
sys.path.insert(0, '/cnic/work/liuql/fuse_opt/DeepSpeed')

def benchmark_single_param(size, warmup=5, steps=50, lr=1e-3, repeat=3):
    """单参数性能测试 - 多次重复取平均"""

    # 收集多次运行结果
    all_results = {'DS_CPUAdam': [], 'FusedAdam_INT8': [], 'FusedAdam_INT4': []}

    for rep in range(repeat):
        results = {}

        # ========== DeepSpeed CPU Adam ==========
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            params_ds = [torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()]
            grad = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01

            opt_ds = DeepSpeedCPUAdam(params_ds, lr=lr, adamw_mode=True)
            params_ds[0].grad = grad

            # Warmup
            for _ in range(warmup):
                opt_ds.step()

            # 计时
            start = time.time()
            for _ in range(steps):
                opt_ds.step()
            elapsed_ds = (time.time() - start) / steps * 1000  # ms per step

            all_results['DS_CPUAdam'].append(elapsed_ds)

        except Exception as e:
            print(f"  [Rep {rep+1}] DeepSpeed CPU Adam 加载失败: {e}")

        # ========== Fused Adam LSQ (INT8) ==========
        try:
            from fused_adam_lsq import FusedAdamLSQ

            params_fused8 = [torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()]
            grad8 = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01

            opt_fused8 = FusedAdamLSQ(params_fused8, lr=lr, q_bits=8)
            params_fused8[0].grad = grad8

            # Warmup
            for _ in range(warmup):
                opt_fused8.step()

            # 计时
            start = time.time()
            for _ in range(steps):
                opt_fused8.step()
            elapsed_fused8 = (time.time() - start) / steps * 1000

            all_results['FusedAdam_INT8'].append(elapsed_fused8)

        except Exception as e:
            print(f"  [Rep {rep+1}] Fused Adam INT8 加载失败: {e}")

        # ========== Fused Adam LSQ (INT4) ==========
        try:
            from fused_adam_lsq import FusedAdamLSQ

            params_fused4 = [torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()]
            grad4 = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01

            opt_fused4 = FusedAdamLSQ(params_fused4, lr=lr, q_bits=4)
            params_fused4[0].grad = grad4

            # Warmup
            for _ in range(warmup):
                opt_fused4.step()

            # 计时
            start = time.time()
            for _ in range(steps):
                opt_fused4.step()
            elapsed_fused4 = (time.time() - start) / steps * 1000

            all_results['FusedAdam_INT4'].append(elapsed_fused4)

        except Exception as e:
            print(f"  [Rep {rep+1}] Fused Adam INT4 加载失败: {e}")

    # 计算平均值和标准差
    avg_results = {}
    import numpy as np
    for key, values in all_results.items():
        if values:
            avg_results[key] = np.mean(values)
            avg_results[f'{key}_std'] = np.std(values)

    # 打印结果
    print(f"  DeepSpeed CPU Adam: {avg_results.get('DS_CPUAdam', 'N/A'):>8.3f}ms/step ± {avg_results.get('DS_CPUAdam_std', 0):.3f}ms ({repeat} reps)")
    print(f"  Fused Adam INT8:    {avg_results.get('FusedAdam_INT8', 'N/A'):>8.3f}ms/step ± {avg_results.get('FusedAdam_INT8_std', 0):.3f}ms")
    print(f"  Fused Adam INT4:    {avg_results.get('FusedAdam_INT4', 'N/A'):>8.3f}ms/step ± {avg_results.get('FusedAdam_INT4_std', 0):.3f}ms")

    return avg_results


def run_benchmark():
    """运行多参数组测试（总参数量18M）"""
    print("=" * 70)
    print("DeepSpeed CPU Adam vs Fused Adam LSQ 性能对比测试")
    print("=" * 70)
    print(f"CPU: ARM Neoverse-V2 (aarch64)")
    print(f"数据类型: BF16")
    print(f"测试参数: warmup=10, steps=100, repeat=5")
    print("=" * 70)

    test_single_18m_param()


def test_single_18m_param():
    print("=" * 70)

    size = 18 * 1024 * 1024 * 4 * 128  # 48GB参数
    print(f"  参数量: {size} elements ({size / 1024 / 1024:.0f}M)")
    print("=" * 70)

    warmup = 2
    steps = 10
    repeat = 5

    import numpy as np

    # Test different group sizes
    for group_size in [128, 1024]:
        print(f"\n--- group_size = {group_size} ---")
        ds_times = []
        fused8_times = []
        fused4_times = []

        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            from fused_adam_lsq import FusedAdamLSQ

            for rep in range(repeat):
                print(f"\n[Rep {rep+1}/{repeat}]")

                # 分块在GPU创建再传CPU（避免GPU显存不足）
                chunk_size = 1 * 1024 * 1024 * 1024  # 1GB per chunk
                chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu()
                    chunks.append(chunk)
                params_ds = torch.cat(chunks).requires_grad_()
                opt_ds = DeepSpeedCPUAdam([params_ds], lr=1e-3, adamw_mode=True)
                # grad也分块创建
                grad_chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu() * 0.01
                    grad_chunks.append(chunk)
                params_ds.grad = torch.cat(grad_chunks)

                for _ in range(warmup):
                    opt_ds.step()

                start = time.time()
                for _ in range(steps):
                    opt_ds.step()
                ds_time = (time.time() - start) / steps * 1000
                ds_times.append(ds_time)
                print(f"  DeepSpeed CPU Adam: {ds_time:.3f}ms/step")

                # Fused Adam INT8 - 分块GPU创建
                chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu()
                    chunks.append(chunk)
                params_fused8 = torch.cat(chunks).requires_grad_()
                opt_fused8 = FusedAdamLSQ([params_fused8], lr=1e-3, q_bits=8, group_size=group_size)
                # grad也分块创建
                grad_chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu() * 0.01
                    grad_chunks.append(chunk)
                params_fused8.grad = torch.cat(grad_chunks)

                for _ in range(warmup):
                    opt_fused8.step()

                start = time.time()
                for _ in range(steps):
                    opt_fused8.step()
                fused8_time = (time.time() - start) / steps * 1000
                fused8_times.append(fused8_time)
                print(f"  Fused Adam INT8:    {fused8_time:.3f}ms/step")

                # Fused Adam INT4 - 分块GPU创建
                chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu()
                    chunks.append(chunk)
                params_fused4 = torch.cat(chunks).requires_grad_()
                opt_fused4 = FusedAdamLSQ([params_fused4], lr=1e-3, q_bits=4, group_size=group_size)
                # grad也分块创建
                grad_chunks = []
                for i in range(0, size, chunk_size):
                    end = min(i + chunk_size, size)
                    chunk = torch.randn(end - i, dtype=torch.bfloat16, device='cuda').cpu() * 0.01
                    grad_chunks.append(chunk)
                params_fused4.grad = torch.cat(grad_chunks)

                for _ in range(warmup):
                    opt_fused4.step()

                start = time.time()
                for _ in range(steps):
                    opt_fused4.step()
                fused4_time = (time.time() - start) / steps * 1000
                fused4_times.append(fused4_time)
                print(f"  Fused Adam INT4:    {fused4_time:.3f}ms/step")

            # 打印汇总结果
            print(f"\n性能汇总 (group_size={group_size}, 5次平均 ± 标准差)")

            ds_avg = np.mean(ds_times)
            ds_std = np.std(ds_times)
            f8_avg = np.mean(fused8_times)
            f8_std = np.std(fused8_times)
            f4_avg = np.mean(fused4_times)
            f4_std = np.std(fused4_times)

            print(f"  DeepSpeed CPU Adam: {ds_avg:.3f} ± {ds_std:.3f} ms/step")
            print(f"  Fused Adam INT8:    {f8_avg:.3f} ± {f8_std:.3f} ms/step")
            print(f"  Fused Adam INT4:    {f4_avg:.3f} ± {f4_std:.3f} ms/step")

            # 计算量化开销
            overhead8 = f8_avg - ds_avg
            overhead8_pct = (f8_avg / ds_avg - 1) * 100
            overhead4 = f4_avg - ds_avg
            overhead4_pct = (f4_avg / ds_avg - 1) * 100

            print(f"  INT8量化额外开销: +{overhead8:.3f}ms ({overhead8_pct:.1f}%)")
            print(f"  INT4量化额外开销: +{overhead4:.3f}ms ({overhead4_pct:.1f}%)")

        except Exception as e:
            print(f"单参数测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()