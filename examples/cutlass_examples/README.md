# CUTLASS

* [CUTLASS](https://zhuanlan.zhihu.com/p/588953452)
* [CUTE](https://zhuanlan.zhihu.com/p/1937220431728845963)
* [reed-cute](https://zhuanlan.zhihu.com/p/661182311)

# Compile-Debug
> rm -rf build
> cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v"