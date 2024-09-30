from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import pybind11
from setuptools import find_packages
import os

# Define the C++ extension
libdesc_include = os.getenv("PREFIX", "/usr/local") + "/include"
libdesc_lib = os.getenv("PREFIX", "/usr/local") + "/lib"

ext_modules = [
    Extension(
        'libdescriptor.libdescriptor',
        ['libdescriptor/python_bindings.cpp',
         'libdescriptor/neighbor/neighbor_list.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            libdesc_include,
        ],
        library_dirs=[
            libdesc_lib,
        ],
        libraries=[
            "descriptor",
        ],
        extra_link_args=[
            "-Wl,-rpath," + libdesc_lib,
        ],
        language='c++'
    ),
]


# Custom build_ext command to include pybind11
class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = ['-std=c++14', '-O3', '-fPIC', '-shared']
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            if sys.platform == 'darwin':
                opts.append('-stdlib=libc++')
                opts.append('-mmacosx-version-min=10.7')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name="libdescriptor",
    version="0.2.5",
    description="Fully differentiable descriptors for atomistic systems",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ase",
        "pybind11",
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    package_data={
        'libdescriptor': [
            "__init__.py",
            "neighbor.py",
            "*.pyi",
        ],
    },
    author="Amit Gupta",
    author_email="gupta839@umn.edu",
    include_package_data=True,
    project_urls={
        "Documentation": "https://libdescriptor.readthedocs.io/",
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/openkim/libdescriptor',
)

