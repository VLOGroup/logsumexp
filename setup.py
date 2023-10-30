from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='logsumexp',
    packages=['logsumexp'],
    package_dir={'logsumexp': './logsumexp'},
    ext_package='logsumexp_cuda_ext',
    ext_modules=[
        CUDAExtension(
            'act',
            sources=['act.cpp', 'act_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O3', '-arch=sm_75']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
