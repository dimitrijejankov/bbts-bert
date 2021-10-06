cmake_minimum_required(VERSION 3.12)

project(pdb)

# set the c++ and cuda version
set(CMAKE_CXX_STANDARD 20)

# for aten
set(CMAKE_PREFIX_PATH third_party/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(bert SHARED bert_types.cc bert_types.h transformer.cc generate_tensor.cc)

find_package(Torch REQUIRED)

# link ATEN
target_link_libraries(bert "${TORCH_LIBRARIES}")
set_property(TARGET bert PROPERTY CXX_STANDARD 14)

add_executable(test_transformer test_transformer.cc)
target_link_libraries(test_transformer bert)

add_executable(generate_commands generate_commands.cc)