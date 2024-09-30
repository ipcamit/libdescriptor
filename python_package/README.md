Libdescriptor 
==============
<img src="https://libdescriptor.readthedocs.io/en/latest/_static/libdescriptor_logo.png" width=200>

[![Documentation Status](https://readthedocs.org/projects/libdescriptor/badge/?version=latest)](https://libdescriptor.readthedocs.io/en/latest/?badge=latest)

Supported Python versions: 3.8, 3.9, 3.10

Libdescriptor is a high performance descriptor library for providing access to fully differentiable descriptor functions.
While `libdescriptor` is a general purpose descriptor library, it's API compatible with KIM models and associated projects.
This will also provide uniform access to various selected descriptors for KLIFF using Pybind11 ports.
For gradient calculations, Libdescriptor relies on [Enzyme AD](https://github.com/EnzymeAD/Enzyme), which provides it with capability to trivially generate near analytical performance gradient functions.
Use of Enzyme AD enables Libdescriptor to not only provide gradients against coordinates, but against hyperparameters as well, thus opening way for better optimized descriptors.
This should enable rapid development, extension and deployment of various descriptors.

<img src="https://libdescriptor.readthedocs.io/en/latest/_images/libdescriptor.svg" width="800">

## Installation
For AMD/Intel based Linux systems, we provide a precompiled binary package for libdescriptor. It can be installed using
```shell
pip install libdescriptor
```

Github Source: [libdescriptor](https://github.com/ipcamit/libdescriptor)

