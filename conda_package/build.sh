# STEP 1: Build Enzyme

export LLVM_DIR=$PREFIX/lib/cmake/llvm

cd $SRC_DIR

cd Enzyme-{{ enzyme_version }}/enzyme
mkdir build
cd build

cmake -GNinja .. -DLLVM_DIR=$LLVM_DIR -DCMAKE_FIND_ROOT_PATH=$PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX
ninja

ninja install

# STEP 2: Build libdescriptor

cd $SRC_DIR
cd libdescriptor
clang++ -shared -fPIC Descriptors.cpp -Xclang -load -Xclang $PREFIX/lib/ClangEnzyme-{{ llvm_major_version }}.so -O3 -o libdescriptor.so
cp libdescriptor.so $PREFIX/lib
cp Descriptors.hpp $PREFIX/include

# STEP 3: Build python package
cd $SRC_DIR
cd python_package
$PYTHON setup.py install
