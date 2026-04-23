import os
import sys

try:
    import torch
except ImportError:
    print("Warning: unable to import torch, please install it if you want to pre-compile any ops.")
else:
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])


class TorchCPUOpBuilder:
    """
    Simplified op builder for JIT compilation via torch.utils.cpp_extension
    """

    def __init__(self, name):
        self.name = name
        self.jit_mode = False
        self.build_for_cpu = False

    def absolute_name(self):
        return f'fused_adam_lsq.{self.name}_op'

    def sources(self):
        return ['csrc/adam/fused_adam_lsq.cpp',
                'csrc/adam/fused_adam_lsq_impl.cpp']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        args = ['-O3', '-std=c++17', '-g', '-Wno-reorder']

        # CPU architecture flags
        args.append('-fopenmp')

        # SIMD flags - detect architecture
        import platform
        machine = platform.machine()

        if machine == 'aarch64' or machine == 'arm64':
            # ARM NEON is always available on aarch64
            args.append('-march=native')
            # NEON is implied in aarch64, no extra flag needed
            print("Detected ARM aarch64 - using NEON SIMD")
        elif machine == 'x86_64' or machine == 'i386' or machine == 'i686':
            args.append('-march=native')
            try:
                from cpuinfo import get_cpu_info
                cpu_info = get_cpu_info()
            except ImportError:
                cpu_info = {'flags': ''}

            if 'avx512' in cpu_info.get('flags', '') or 'avx512f' in cpu_info.get('flags', ''):
                args.append('-D__AVX512__')
            elif 'avx2' in cpu_info.get('flags', ''):
                args.append('-D__AVX256__')
        else:
            # Unknown architecture, use scalar
            args.append('-march=native')

        return args

    def extra_ldflags(self):
        return ['-fopenmp']

    def builder(self):
        from torch.utils.cpp_extension import CppExtension
        include_dirs = [os.path.abspath(x) for x in self.include_paths()]
        return CppExtension(name=self.absolute_name(),
                            sources=self.sources(),
                            include_dirs=include_dirs,
                            extra_compile_args={'cxx': self.cxx_args()},
                            extra_link_args=self.extra_ldflags())

    def load(self, verbose=False):
        from torch.utils.cpp_extension import load

        sources = [os.path.abspath(path) for path in self.sources()]
        extra_include_paths = [os.path.abspath(path) for path in self.include_paths()]

        op_module = load(name=self.name,
                         sources=sources,
                         extra_include_paths=extra_include_paths,
                         extra_cflags=self.cxx_args(),
                         extra_ldflags=self.extra_ldflags(),
                         verbose=verbose)

        return op_module


class FusedAdamLSQBuilder(TorchCPUOpBuilder):
    """
    Builder for Fused Adam + LSQ Quantization optimizer
    """
    BUILD_VAR = "DS_BUILD_FUSED_ADAM_LSQ"
    NAME = "fused_adam_lsq"

    def __init__(self):
        super().__init__(name=self.NAME)

    def sources(self):
        return ['csrc/adam/fused_adam_lsq.cpp',
                'csrc/adam/fused_adam_lsq_impl.cpp']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        args = super().cxx_args()
        # Add BF16 support flag for AVX512
        args.append('-DBF16_AVAILABLE')
        return args