cmake_minimum_required(VERSION 3.26)
project(rabitq_search LANGUAGES CXX CUDA)

# Point CMAKE_PREFIX_PATH at where you installed the cuVS fork (conda prefix or
# third-party/cuvs_rabitq/cpp/build/install). find_package pulls raft/rmm transitively.
find_package(cuvs REQUIRED)

add_executable(rabitq_search rabitq_search.cu)
set_target_properties(rabitq_search PROPERTIES
    CUDA_ARCHITECTURES "70"          # V100 = sm_70 ; RTX 5090 = "120"
    CXX_STANDARD 17
    CUDA_STANDARD 17)
target_link_libraries(rabitq_search PRIVATE cuvs::cuvs)