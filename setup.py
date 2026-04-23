from setuptools import setup, find_packages

setup(
    name='fused_adam_lsq',
    version='0.1.0',
    description='Fused Adam optimizer with LSQ quantization for ZeRO-Offload',
    author='me',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'torch>=1.9.0',
        'py-cpuinfo',
    ],
    python_requires='>=3.7',
)