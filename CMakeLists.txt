cmake_minimum_required(VERSION 3.12)

project(pdb)

# set the c++ and cuda version
set(CMAKE_CXX_STANDARD 20)

# do -O3 and set the instruction set to the native one for this machine
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

# some stuff in the linker to make .so loading nicer
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -ldl" )
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -ldl")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -ldl")

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")

# for aten
set(CMAKE_PREFIX_PATH third_party/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(bert SHARED bert_types.cc 
                        bert_types.h
                        transformer.cc 
                        generate_tensor.cc 
                        bert_lib_init.cc)

find_package(Torch REQUIRED)

# link ATEN
target_link_libraries(bert "${TORCH_LIBRARIES}")
set_property(TARGET bert PROPERTY CXX_STANDARD 14)

add_executable(test_transformer test_transformer.cc
                                bert_types.cc 
                                bert_types.h
                                transformer.cc 
                                generate_tensor.cc)

# link ATEN
target_link_libraries(test_transformer "${TORCH_LIBRARIES}")
set_property(TARGET test_transformer PROPERTY CXX_STANDARD 14)

add_executable(generate_commands generate_commands.cc)