import torch
from .op_builder import FusedAdamLSQBuilder


class FusedAdamLSQ(torch.optim.Optimizer):
    """
    Fused Adam optimizer with LSQ quantization for ZeRO-Offload.

    This optimizer performs Adam weight update on CPU and immediately
    quantizes the updated weights to INT4/INT8, storing the quantized
    data in a separate buffer while keeping high-precision weights
    for backward propagation.

    The fusion allows the quantization to happen while updated weights
    are still in CPU L1 cache, reducing memory access overhead.

    **无状态架构：C++端不存储任何量化参数，每次step传入所有tensor。**
    **必须手动调用set方法设置所有buffer才能使用step()。**

    Arguments:
        model_params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: 1e-3)
        betas (tuple): coefficients for computing running averages (default: (0.9, 0.999))
        eps (float): term added to denominator for numerical stability (default: 1e-8)
        weight_decay (float): weight decay coefficient (default: 0)
        bias_correction (bool): whether to apply bias correction (default: True)
        adamw_mode (bool): use AdamW if True, otherwise Adam (default: True)
        fp32_optimizer_states (bool): use FP32 for optimizer states (default: True)
        group_size (int): number of elements per quantization group (default: 128)
        q_bits (int): quantization bits, 4 or 8 (default: 8)
    """

    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 adamw_mode=True,
                 fp32_optimizer_states=True,
                 group_size=128,
                 q_bits=8):
        if q_bits not in [4, 8]:
            raise ValueError(f"q_bits must be 4 or 8, got {q_bits}")

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction)

        super(FusedAdamLSQ, self).__init__(model_params, default_args)

        self.opt_id = FusedAdamLSQ.optimizer_id
        FusedAdamLSQ.optimizer_id = FusedAdamLSQ.optimizer_id + 1

        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.group_size = group_size
        self.default_q_bits = q_bits  # 默认值，可per-param覆盖

        # Per-parameter quantization settings
        self._param_q_bits = {}  # per-parameter q_bits (4, 8, or None for BF16)

        # Load the C++ extension
        self.ds_opt_adam_lsq = FusedAdamLSQBuilder().load()

        self.ds_opt_adam_lsq.create_adam_lsq(
            self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, True
        )

        # Per-parameter quantization buffers (Python端状态，需手动设置)
        self.quant_buffers = {}    # Quantized weight buffers (uint8)
        self.delta_tensors = {}    # Per-group scale parameters
        self.z_tensors = {}        # Per-group zero point parameters
        self._param_meta = {}      # 参数元数据

    def __del__(self):
        try:
            self.ds_opt_adam_lsq.destroy_adam_lsq(self.opt_id)
        except AttributeError:
            pass

    def _init_meta(self, p, q_bits=None):
        """初始化参数元数据（仅计算尺寸信息）。

        Args:
            p: 参数tensor
            q_bits: 量化位数，如果None则使用default_q_bits
        """
        if p in self._param_meta:
            return

        num_elements = p.numel()
        num_groups = num_elements // self.group_size
        if num_elements % self.group_size != 0:
            num_groups += 1

        # Quantized buffer size: INT8 needs 1 byte per element, INT4 needs 1 byte per 2 elements
        # Use provided q_bits or default
        effective_q_bits = q_bits if q_bits is not None else self.default_q_bits
        if effective_q_bits == 8:
            quant_size = num_elements
        else:
            quant_size = num_elements // 2 + (num_elements % 2)

        self._param_meta[p] = {
            'num_elements': num_elements,
            'num_groups': num_groups,
            'quant_size': quant_size,
        }

    def get_delta_tensor(self, p):
        """Get delta tensor for parameter p."""
        return self.delta_tensors.get(p)

    def get_z_tensor(self, p):
        """Get z tensor for parameter p."""
        return self.z_tensors.get(p)

    def get_quant_buffer(self, p):
        """Get quantized buffer for parameter p."""
        return self.quant_buffers.get(p)

    def get_param_meta(self, p):
        """Get parameter metadata (num_elements, num_groups, quant_size)."""
        if p not in self._param_meta:
            self._init_meta(p)
        return self._param_meta.get(p)

    def set_q_bits(self, p, q_bits):
        """Set q_bits for a specific parameter.

        Args:
            p: parameter tensor
            q_bits: 4, 8, or None (BF16, no quantization)
        """
        if q_bits is not None and q_bits not in [4, 8]:
            raise ValueError(f"q_bits must be 4, 8, or None, got {q_bits}")
        self._param_q_bits[p] = q_bits

    def get_q_bits(self, p):
        """Get q_bits for a parameter, returns default if not set."""
        return self._param_q_bits.get(p, self.default_q_bits)

    def is_quantized(self, p):
        """Check if a parameter should be quantized.

        A parameter is quantized if:
        1. It has a quant_buffer set (explicitly quantized)
        2. OR its q_bits is set to 4 or 8

        Returns:
            True if parameter should use quantized kernel
            False if parameter should use non-quantized kernel (BF16)
        """
        # Check explicit quant_buffer
        if p in self.quant_buffers and self.quant_buffers[p] is not None:
            return True
        # Check q_bits setting
        q_bits = self.get_q_bits(p)
        return q_bits in [4, 8]

    def set_quant_buffer(self, p, quant_buffer):
        """Set quantized buffer for parameter p.

        Args:
            p: parameter tensor
            quant_buffer: uint8 tensor for quantized data
        """
        # Use the parameter's q_bits for meta initialization
        q_bits = self.get_q_bits(p)
        if p not in self._param_meta:
            self._init_meta(p, q_bits)
        else:
            # Recalculate quant_size if q_bits changed
            num_elements = p.numel()
            if q_bits == 8:
                quant_size = num_elements
            else:
                quant_size = num_elements // 2 + (num_elements % 2)
            self._param_meta[p]['quant_size'] = quant_size

        meta = self._param_meta[p]
        if quant_buffer.numel() != meta['quant_size']:
            raise ValueError(f"quant_buffer size {quant_buffer.numel()} != expected {meta['quant_size']} (q_bits={q_bits})")
        if quant_buffer.dtype != torch.uint8:
            raise ValueError(f"quant_buffer must be uint8, got {quant_buffer.dtype}")

        self.quant_buffers[p] = quant_buffer

    def set_delta_tensor(self, p, delta):
        """Set delta tensor (scale parameter) for parameter p.

        Args:
            p: parameter tensor
            delta: float32 tensor, size = num_groups
        """
        if p not in self._param_meta:
            self._init_meta(p)

        meta = self._param_meta[p]
        if delta.numel() != meta['num_groups']:
            raise ValueError(f"delta size {delta.numel()} != expected {meta['num_groups']}")
        if delta.dtype != torch.float32:
            raise ValueError(f"delta must be float32, got {delta.dtype}")

        self.delta_tensors[p] = delta

    def set_z_tensor(self, p, z):
        """Set z tensor (zero point parameter) for parameter p.

        Args:
            p: parameter tensor
            z: float32 tensor, size = num_groups
        """
        if p not in self._param_meta:
            self._init_meta(p)

        meta = self._param_meta[p]
        if z.numel() != meta['num_groups']:
            raise ValueError(f"z size {z.numel()} != expected {meta['num_groups']}")
        if z.dtype != torch.float32:
            raise ValueError(f"z must be float32, got {z.dtype}")

        self.z_tensors[p] = z

    @torch.no_grad()
    def step(self, closure=None):
        """Update model parameters with fused Adam + LSQ quantization.

        支持混合精度：
        - 量化参数 (INT4/INT8): 需要设置 quant_buffer, delta, z，调用 adam_update_lsq
        - BF16参数: 不设置 quant_buffer，调用 adam_update (无量化)

        Args:
            closure (callable, optional): closure to compute loss

        Returns:
            loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                assert p.device == device, \
                    f"FusedAdamLSQ param is on {p.device} and must be 'cpu'"

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

                state['step'] += 1
                beta1, beta2 = group['betas']

                # 根据参数是否需要量化选择kernel
                if self.is_quantized(p):
                    # INT4/INT8: 使用量化kernel
                    # 验证量化参数已设置
                    if p not in self.quant_buffers:
                        raise ValueError(f"quant_buffer not set for quantized parameter. Call set_quant_buffer() first.")
                    if p not in self.delta_tensors:
                        raise ValueError(f"delta tensor not set for quantized parameter. Call set_delta_tensor() first.")
                    if p not in self.z_tensors:
                        raise ValueError(f"z tensor not set for quantized parameter. Call set_z_tensor() first.")

                    q_bits = self.get_q_bits(p)
                    if q_bits not in [4, 8]:
                        raise ValueError(f"Invalid q_bits {q_bits} for quantized parameter, must be 4 or 8")

                    # Call fused C++ kernel with quantization
                    self.ds_opt_adam_lsq.adam_update_lsq(
                        self.opt_id,
                        state['step'],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['weight_decay'],
                        group['bias_correction'],
                        p.data,
                        p.grad.data,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        self.quant_buffers[p],
                        self.delta_tensors[p],
                        self.z_tensors[p],
                        self.group_size,
                        q_bits  # 使用per-param q_bits
                    )
                else:
                    # BF16: 使用无量化kernel
                    self.ds_opt_adam_lsq.adam_update(
                        self.opt_id,
                        state['step'],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['weight_decay'],
                        group['bias_correction'],
                        p.data,
                        p.grad.data,
                        state['exp_avg'],
                        state['exp_avg_sq']
                    )

        return loss

    @torch.no_grad()
    def step_without_quant(self, closure=None):
        """Update model parameters with Adam only (no quantization).

        不需要设置量化buffer，只执行Adam更新。

        Args:
            closure (callable, optional): closure to compute loss

        Returns:
            loss if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                assert p.device == device, \
                    f"FusedAdamLSQ param is on {p.device} and must be 'cpu'"

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

                state['step'] += 1
                beta1, beta2 = group['betas']

                # Call Adam-only C++ kernel (no quantization)
                self.ds_opt_adam_lsq.adam_update(
                    self.opt_id,
                    state['step'],
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    group['weight_decay'],
                    group['bias_correction'],
                    p.data,
                    p.grad.data,
                    state['exp_avg'],
                    state['exp_avg_sq']
                )

        return loss

    # def update_quant_params(self, p, delta_grad, z_grad, lr_quant=1e-4):
    #     """Update delta and z tensors using gradients (for LSQ learning).

    #     Args:
    #         p: parameter whose quantization params to update
    #         delta_grad: gradient for delta tensor
    #         z_grad: gradient for z tensor
    #         lr_quant: learning rate for quantization parameters
    #     """
    #     if p in self.delta_tensors:
    #         self.delta_tensors[p].data -= lr_quant * delta_grad
    #         self.delta_tensors[p].data.clamp_(min=1e-6)
    #         self.z_tensors[p].data -= lr_quant * z_grad


def dequantize_weight(quant_buffer, delta, z, q_bits, group_size, original_size, dtype=torch.bfloat16):
    """
    Dequantize INT4/INT8 weights back to high precision.

    Args:
        quant_buffer: uint8 tensor containing quantized weights
        delta: per-group scale tensor
        z: per-group zero point tensor
        q_bits: 4 or 8
        group_size: elements per group
        original_size: number of elements in original weight tensor
        dtype: output dtype (default: bfloat16)

    Returns:
        Dequantized weight tensor
    """
    output = torch.empty(original_size, dtype=dtype)

    if q_bits == 8:
        for i in range(original_size):
            group_id = i // group_size
            q = quant_buffer[i].item()
            output[i] = q * delta[group_id] + z[group_id]
    else:
        for i in range(original_size):
            group_id = i // group_size
            packed_idx = i // 2
            packed = quant_buffer[packed_idx].item()
            if i % 2 == 0:
                q = (packed >> 4) & 0x0F
            else:
                q = packed & 0x0F
            output[i] = q * delta[group_id] + z[group_id]

    return output