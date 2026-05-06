#!/usr/bin/env python
"""
性能对比测试：DeepSpeed CPU Adam vs Fused Adam LSQ

测试目标：
1. DeepSpeed CPU Adam（无量化基准）
2. FusedAdamLSQ step_without_quant（无量化）
3. FusedAdamLSQ step（带LSQ量化）
4. 量化开销分析

新架构说明：
- Python端状态管理 + C++无状态架构
- 必须手动调用set方法设置所有buffer才能使用step()
- step_without_quant不需要设置buffer
"""

import torch
import time
import sys
import os

sys.path.insert(0, '/cnic/work/liuql/fuse_opt/DeepSpeed')


def benchmark_quant_vs_no_quant():
    """对比量化与不量化的性能差距"""
    print("\n" + "=" * 70)
    print("量化 vs 不量化 性能对比")
    print("=" * 70)

    sizes = [1024, 10*1024, 100*1024, 1024*1024]
    warmup = 3
    steps = 30

    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        from fused_adam_lsq import FusedAdamLSQ

        print(f"\n{'Size':>12} {'DS Adam':>12} {'Fused(noQ)':>12} {'Fused(INT8)':>12} {'Fused(INT4)':>12} {'Q开销8':>10} {'Q开销4':>10}")
        print("-" * 80)

        for size in sizes:
            group_size = 128
            num_groups = size // group_size
            results = {}

            # DeepSpeed CPU Adam（基准）
            params_ds = torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()
            grad_ds = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01
            opt_ds = DeepSpeedCPUAdam([params_ds], lr=1e-3, adamw_mode=True)
            params_ds.grad = grad_ds

            for _ in range(warmup):
                opt_ds.step()
            start = time.time()
            for _ in range(steps):
                opt_ds.step()
            results['ds'] = (time.time() - start) / steps * 1000

            # FusedAdamLSQ step_without_quant（无量化）
            params_noq = torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()
            grad_noq = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01
            opt_noq = FusedAdamLSQ([params_noq], lr=1e-3, q_bits=8)
            params_noq.grad = grad_noq

            for _ in range(warmup):
                opt_noq.step_without_quant()
            start = time.time()
            for _ in range(steps):
                opt_noq.step_without_quant()
            results['noq'] = (time.time() - start) / steps * 1000

            # FusedAdamLSQ step（INT8量化）- 需手动设置buffer
            params_q8 = torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()
            grad_q8 = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01
            opt_q8 = FusedAdamLSQ([params_q8], lr=1e-3, q_bits=8)
            params_q8.grad = grad_q8

            # 手动设置buffer
            quant_buffer8 = torch.zeros(size, dtype=torch.uint8)
            delta8 = torch.ones(num_groups, dtype=torch.float32)
            z8 = torch.zeros(num_groups, dtype=torch.float32)
            opt_q8.set_quant_buffer(params_q8, quant_buffer8)
            opt_q8.set_delta_tensor(params_q8, delta8)
            opt_q8.set_z_tensor(params_q8, z8)

            for _ in range(warmup):
                opt_q8.step()
            start = time.time()
            for _ in range(steps):
                opt_q8.step()
            results['q8'] = (time.time() - start) / steps * 1000

            # FusedAdamLSQ step（INT4量化）
            params_q4 = torch.randn(size, dtype=torch.bfloat16, device='cpu').requires_grad_()
            grad_q4 = torch.randn(size, dtype=torch.bfloat16, device='cpu') * 0.01
            opt_q4 = FusedAdamLSQ([params_q4], lr=1e-3, q_bits=4)
            params_q4.grad = grad_q4

            # 手动设置buffer
            quant_buffer4 = torch.zeros(size // 2, dtype=torch.uint8)
            delta4 = torch.ones(num_groups, dtype=torch.float32)
            z4 = torch.zeros(num_groups, dtype=torch.float32)
            opt_q4.set_quant_buffer(params_q4, quant_buffer4)
            opt_q4.set_delta_tensor(params_q4, delta4)
            opt_q4.set_z_tensor(params_q4, z4)

            for _ in range(warmup):
                opt_q4.step()
            start = time.time()
            for _ in range(steps):
                opt_q4.step()
            results['q4'] = (time.time() - start) / steps * 1000

            # 计算量化开销
            overhead8 = results['q8'] - results['noq']
            overhead8_pct = overhead8 / results['noq'] * 100 if results['noq'] > 0 else 0
            overhead4 = results['q4'] - results['noq']
            overhead4_pct = overhead4 / results['noq'] * 100 if results['noq'] > 0 else 0

            print(f"{size:>12} {results['ds']:>10.3f}ms {results['noq']:>10.3f}ms {results['q8']:>10.3f}ms {results['q4']:>10.3f}ms {overhead8:>8.3f}ms({overhead8_pct:>4.1f}%) {overhead4:>8.3f}ms({overhead4_pct:>4.1f}%)")

        print("-" * 80)
        print("注: Q开销 = Fused(量化) - Fused(无量化)")

    except Exception as e:
        print(f"对比测试失败: {e}")
        import traceback
        traceback.print_exc()


def benchmark_large_scale():
    """大规模参数测试"""
    print("\n" + "=" * 70)
    print("大规模参数性能测试")
    print("=" * 70)

    size = 18 * 1024 * 1024  # 100M elements
    print(f"参数量: {size} elements ({size / 1024 / 1024:.0f}MB)")

    warmup = 20
    steps = 50
    repeat = 5

    import numpy as np

    for group_size in [128, 1024]:
        print(f"\n--- group_size = {group_size} ---")
        print(f"\n{'Rep':>6} {'DS Adam':>12} {'Fused(noQ)':>12} {'Fused(INT8)':>12} {'Fused(INT4)':>12}")

        ds_times = []
        noq_times = []
        q8_times = []
        q4_times = []

        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            from fused_adam_lsq import FusedAdamLSQ

            num_groups = size // group_size

            for rep in range(repeat):
                # DeepSpeed CPU Adam - GPU初始化后传回CPU
                params_ds = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu().requires_grad_()
                opt_ds = DeepSpeedCPUAdam([params_ds], lr=1e-3, adamw_mode=True)
                params_ds.grad = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu() * 0.01

                for _ in range(warmup):
                    opt_ds.step()
                start = time.time()
                for _ in range(steps):
                    opt_ds.step()
                ds_time = (time.time() - start) / steps * 1000
                ds_times.append(ds_time)

                # FusedAdamLSQ step_without_quant
                params_noq = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu().requires_grad_()
                opt_noq = FusedAdamLSQ([params_noq], lr=1e-3, q_bits=8, group_size=group_size)
                params_noq.grad = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu() * 0.01

                for _ in range(warmup):
                    opt_noq.step_without_quant()
                start = time.time()
                for _ in range(steps):
                    opt_noq.step_without_quant()
                noq_time = (time.time() - start) / steps * 1000
                noq_times.append(noq_time)

                # FusedAdamLSQ INT8 - 手动设置buffer
                params_q8 = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu().requires_grad_()
                opt_q8 = FusedAdamLSQ([params_q8], lr=1e-3, q_bits=8, group_size=group_size)
                params_q8.grad = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu() * 0.01

                quant_buffer8 = torch.zeros(size, dtype=torch.uint8)
                delta8 = torch.ones(num_groups, dtype=torch.float32)
                z8 = torch.zeros(num_groups, dtype=torch.float32)
                opt_q8.set_quant_buffer(params_q8, quant_buffer8)
                opt_q8.set_delta_tensor(params_q8, delta8)
                opt_q8.set_z_tensor(params_q8, z8)

                for _ in range(warmup):
                    opt_q8.step()
                start = time.time()
                for _ in range(steps):
                    opt_q8.step()
                q8_time = (time.time() - start) / steps * 1000
                q8_times.append(q8_time)

                # FusedAdamLSQ INT4
                params_q4 = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu().requires_grad_()
                opt_q4 = FusedAdamLSQ([params_q4], lr=1e-3, q_bits=4, group_size=group_size)
                params_q4.grad = torch.randn(size, dtype=torch.bfloat16, device='cuda').cpu() * 0.01

                quant_buffer4 = torch.zeros(size // 2, dtype=torch.uint8)
                delta4 = torch.ones(num_groups, dtype=torch.float32)
                z4 = torch.zeros(num_groups, dtype=torch.float32)
                opt_q4.set_quant_buffer(params_q4, quant_buffer4)
                opt_q4.set_delta_tensor(params_q4, delta4)
                opt_q4.set_z_tensor(params_q4, z4)

                for _ in range(warmup):
                    opt_q4.step()
                start = time.time()
                for _ in range(steps):
                    opt_q4.step()
                q4_time = (time.time() - start) / steps * 1000
                q4_times.append(q4_time)

                print(f"{rep+1:>6}/{repeat} {ds_time:>10.3f}ms {noq_time:>10.3f}ms {q8_time:>10.3f}ms {q4_time:>10.3f}ms")

            # 汇总统计（带异常值去除）
            print(f"\n{'='*60}")
            print(f"性能汇总 (group_size={group_size}, {repeat}次测量)")
            print(f"{'='*60}")

            # 异常值去除函数（IQR方法）
            def remove_outliers_iqr(times):
                """使用 IQR 方法去除异常值"""
                times_sorted = sorted(times)
                n = len(times_sorted)
                if n < 4:
                    return times_sorted, []

                q1_idx = n // 4
                q3_idx = (3 * n) // 4
                q1 = times_sorted[q1_idx]
                q3 = times_sorted[q3_idx]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                normal = [t for t in times_sorted if lower_bound <= t <= upper_bound]
                outliers = [t for t in times_sorted if t < lower_bound or t > upper_bound]
                return normal, outliers

            # 原始数据统计
            ds_raw_avg = np.mean(ds_times)
            noq_raw_avg = np.mean(noq_times)
            q8_raw_avg = np.mean(q8_times)
            q4_raw_avg = np.mean(q4_times)

            # 去除异常值
            ds_normal, ds_outliers = remove_outliers_iqr(ds_times)
            noq_normal, noq_outliers = remove_outliers_iqr(noq_times)
            q8_normal, q8_outliers = remove_outliers_iqr(q8_times)
            q4_normal, q4_outliers = remove_outliers_iqr(q4_times)

            # 打印异常值信息
            if ds_outliers or noq_outliers or q8_outliers or q4_outliers:
                print(f"\n异常值检测结果:")
                if ds_outliers:
                    print(f"  DeepSpeed: 原始{len(ds_times)}个, 去除{len(ds_outliers)}个异常值 {[f'{v:.3f}ms' for v in ds_outliers]}")
                if noq_outliers:
                    print(f"  Fused(noQ): 去除{len(noq_outliers)}个异常值 {[f'{v:.3f}ms' for v in noq_outliers]}")
                if q8_outliers:
                    print(f"  Fused(INT8): 去除{len(q8_outliers)}个异常值 {[f'{v:.3f}ms' for v in q8_outliers]}")
                if q4_outliers:
                    print(f"  Fused(INT4): 去除{len(q4_outliers)}个异常值 {[f'{v:.3f}ms' for v in q4_outliers]}")

            # 使用去除异常值后的数据计算平均值
            ds_avg = np.mean(ds_normal) if ds_normal else ds_raw_avg
            noq_avg = np.mean(noq_normal) if noq_normal else noq_raw_avg
            q8_avg = np.mean(q8_normal) if q8_normal else q8_raw_avg
            q4_avg = np.mean(q4_normal) if q4_normal else q4_raw_avg

            # 有效样本数
            ds_n = len(ds_normal)
            noq_n = len(noq_normal)
            q8_n = len(q8_normal)
            q4_n = len(q4_normal)

            print(f"\n有效样本数: DS={ds_n}, noQ={noq_n}, INT8={q8_n}, INT4={q4_n}")

            # 计算标准差
            ds_std = np.std(ds_normal) if len(ds_normal) > 1 else 0
            noq_std = np.std(noq_normal) if len(noq_normal) > 1 else 0
            q8_std = np.std(q8_normal) if len(q8_normal) > 1 else 0
            q4_std = np.std(q4_normal) if len(q4_normal) > 1 else 0

            print(f"\n  DeepSpeed CPU Adam:     {ds_avg:.3f} ± {ds_std:.3f} ms/step (原始: {ds_raw_avg:.3f})")
            print(f"  FusedAdam (no quant):   {noq_avg:.3f} ± {noq_std:.3f} ms/step (原始: {noq_raw_avg:.3f})")
            print(f"  FusedAdam (INT8 quant): {q8_avg:.3f} ± {q8_std:.3f} ms/step (原始: {q8_raw_avg:.3f})")
            print(f"  FusedAdam (INT4 quant): {q4_avg:.3f} ± {q4_std:.3f} ms/step (原始: {q4_raw_avg:.3f})")

            # 与DeepSpeed对比
            print(f"\n与DeepSpeed CPU Adam对比:")
            noq_vs_ds = (noq_avg - ds_avg) / ds_avg * 100
            q8_vs_ds = (q8_avg - ds_avg) / ds_avg * 100
            q4_vs_ds = (q4_avg - ds_avg) / ds_avg * 100
            print(f"  FusedAdam (no quant):   {noq_vs_ds:+.1f}% vs DeepSpeed")
            print(f"  FusedAdam (INT8 quant): {q8_vs_ds:+.1f}% vs DeepSpeed")
            print(f"  FusedAdam (INT4 quant): {q4_vs_ds:+.1f}% vs DeepSpeed")

            # 量化开销分析
            print(f"\n量化开销分析:")
            overhead8 = q8_avg - noq_avg
            overhead8_pct = overhead8 / noq_avg * 100
            overhead4 = q4_avg - noq_avg
            overhead4_pct = overhead4 / noq_avg * 100
            print(f"  INT8量化开销: +{overhead8:.3f}ms ({overhead8_pct:+.1f}%)")
            print(f"  INT4量化开销: +{overhead4:.3f}ms ({overhead4_pct:+.1f}%)")

        except Exception as e:
            print(f"大规模测试失败: {e}")
            import traceback
            traceback.print_exc()


def run_benchmark():
    """运行所有benchmark测试"""
    print("=" * 70)
    print("DeepSpeed CPU Adam vs Fused Adam LSQ 性能对比测试")
    print("=" * 70)
    print("架构: Python端状态管理 + C++无状态")
    print("必须手动设置buffer才能使用step()")
    print("=" * 70)

    # 1. 量化 vs 不量化对比
    # benchmark_quant_vs_no_quant()

    # 2. 大规模测试
    try:
        benchmark_large_scale()
    except Exception as e:
        print(f"大规模测试跳过: {e}")

    print("\n" + "=" * 70)
    print("Benchmark完成")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()