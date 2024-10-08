# STEP 1: Build Enzyme
set -e
set -x

echo "llvm_major_version: $llvm_major_version"
export LLVM_DIR=$PREFIX/lib/cmake/llvm

# 1. Build Enzyme
cd $SRC_DIR

cd Enzyme-$enzyme_version/enzyme
mkdir build
cd build

cmake -GNinja .. -DLLVM_DIR=$LLVM_DIR -DCMAKE_FIND_ROOT_PATH=$PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX
ninja
ninja install

# test example
cat > test.cpp <<EOF
#include <cstdio>
#include <iostream>
#include <map>

double square(double x) {
    return x * x;
}

double __enzyme_autodiff(void*, double);
int main() {
    double x = 3.14;
    // Evaluates to 2 * x = 6.28
    double grad_x = __enzyme_autodiff((void*)square, x);
    printf("square'(%f) \n", grad_x);
}

EOF

clang++ -Xclang  -load -Xclang `pwd`/Enzyme/ClangEnzyme-$llvm_major_version.so -I$BUILD_PREFIX/include test.cpp -O3 -o test.x
./test.x

echo "--- Enzyme build successful ---"

export ENZYME_PATH=`pwd`/Enzyme/ClangEnzyme-$llvm_major_version.so

# STEP 2: Build libdescriptor
cd $SRC_DIR
cd libdescriptor
clang++ -shared -fPIC Descriptors.cpp -Xclang -load -Xclang $ENZYME_PATH  -I$BUILD_PREFIX/include/eigen3 -I$BUILD_PREFIX/include -O3 -o libdescriptor.so
cp libdescriptor.so $PREFIX/lib
cp Descriptors.hpp $PREFIX/include

# STEP 3: Build python package
cd python_package
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
