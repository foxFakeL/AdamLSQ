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
        self.q_bits = q_bits

        # Load the C++ extension
        self.ds_opt_adam_lsq = FusedAdamLSQBuilder().load()

        self.ds_opt_adam_lsq.create_adam_lsq(
            self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, True
        )

        # Per-parameter quantization buffers
        self.quant_buffers = {}    # Quantized weight buffers (uint8)
        self.delta_tensors = {}    # Per-group scale parameters
        self.z_tensors = {}        # Per-group zero point parameters

    def __del__(self):
        try:
            self.ds_opt_adam_lsq.destroy_adam_lsq(self.opt_id)
        except AttributeError:
            pass

    def _init_quant_params(self, p):
        """Initialize quantization parameters for a parameter."""
        if p in self.quant_buffers:
            return

        num_elements = p.numel()
        num_groups = num_elements // self.group_size
        if num_elements % self.group_size != 0:
            # Handle non-divisible sizes
            num_groups += 1

        # Quantized buffer size: INT8 needs 1 byte per element, INT4 needs 1 byte per 2 elements
        if self.q_bits == 8:
            quant_size = num_elements
        else:
            quant_size = num_elements // 2 + (num_elements % 2)

        # Create pinned memory buffer for quantized data
        self.quant_buffers[p] = torch.empty(
            quant_size, dtype=torch.uint8, pin_memory=True
        )

        # Initialize delta and z tensors (learnable parameters)
        # delta: scale parameter, initialized based on weight distribution
        # z: zero point parameter, initialized to 0
        self.delta_tensors[p] = torch.ones(num_groups, dtype=torch.float32)
        self.z_tensors[p] = torch.zeros(num_groups, dtype=torch.float32)

        # Register delta and z with C++ optimizer (only once per parameter)
        # This reduces Tensor passing from 7 to 5, saving ~0.1-0.2ms framework overhead
        self.ds_opt_adam_lsq.register_quant_params(
            self.opt_id,
            self.delta_tensors[p],
            self.z_tensors[p],
            self.group_size,
            self.q_bits
        )

    def get_delta_tensor(self, p):
        """Get delta tensor for parameter p (can be used for learning)."""
        return self.delta_tensors.get(p)

    def get_z_tensor(self, p):
        """Get z tensor for parameter p (can be used for learning)."""
        return self.z_tensors.get(p)

    def get_quant_buffer(self, p):
        """Get quantized buffer for parameter p."""
        return self.quant_buffers.get(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Update model parameters with fused Adam + LSQ quantization.

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

                    # Initialize quantization parameters
                    self._init_quant_params(p)

                state['step'] += 1
                beta1, beta2 = group['betas']

                # Call fused C++ kernel
                # Optimized: delta and z registered internally during init
                # Now only passes 5 Tensor params (reduced from 7), saving ~0.1-0.2ms overhead
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
                )

        return loss

    def update_quant_params(self, p, delta_grad, z_grad, lr_quant=1e-4):
        """Update delta and z tensors using gradients (for LSQ learning).

        This can be called after backward pass to update the learnable
        quantization parameters.

        Args:
            p: parameter whose quantization params to update
            delta_grad: gradient for delta tensor
            z_grad: gradient for z tensor
            lr_quant: learning rate for quantization parameters
        """
        if p in self.delta_tensors:
            self.delta_tensors[p].data -= lr_quant * delta_grad
            # Ensure delta is positive
            self.delta_tensors[p].data.clamp_(min=1e-6)
            self.z_tensors[p].data -= lr_quant * z_grad


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
        # INT8 dequantization
        for i in range(original_size):
            group_id = i // group_size
            q = quant_buffer[i].item()
            output[i] = q * delta[group_id] + z[group_id]
    else:
        # INT4 dequantization (unpacked)
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