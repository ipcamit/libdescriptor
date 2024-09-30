clang++ -shared -fPIC Descriptors.cpp -Xclang -load -Xclang /opt/enzyme/enzyme/build/Enzyme/ClangEnzyme-13.so -O3 -o libdescriptor.so
