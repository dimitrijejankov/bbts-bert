cmake_minimum_required(VERSION 3.12)

project(pdb)

# for aten
set(CMAKE_PREFIX_PATH third_party/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(generate test_transformer.cc bert_types.cc bert_types.h transformer.cc)

find_package(Torch REQUIRED)

# process lib
target_link_libraries(generate "${TORCH_LIBRARIES}")
set_property(TARGET generate PROPERTY CXX_STANDARD 14)