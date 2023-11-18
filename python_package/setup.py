from distutils.sysconfig import get_config_vars
from setuptools import find_packages, setup

setup(
    name="libdescriptor",
    version="0.0.7",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ase",
        "pybind11",
    ],
    package_data={
        'libdescriptor': [
                        "__init__.py",
                        "neighbor.py",
                        "libc.so.6",
                        "libdescriptor.cpython-39-x86_64-linux-gnu.so",
                        "libdescriptor.so",
                        "libgcc_s.so.1",
                        "libm.so.6",
                        "libstdc++.so.6"
                        ],
    },
    author="Amit Gupta",
    include_package_data=True,
)