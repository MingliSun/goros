cmake_minimum_required(VERSION 3.17)
project(autoFusion)

set(CMAKE_CXX_STANDARD 17)
#add_definitions("-Wall -g")
include_directories("include")
AUX_SOURCE_DIRECTORY(src/core CORE_CPP)
AUX_SOURCE_DIRECTORY(src/operators OPERATORS_CPP)
AUX_SOURCE_DIRECTORY(tests TESTS_CPP)
#AUX_SOURCE_DIRECTORY()
add_executable(autoFusion main.cpp ${TESTS_CPP} ${CORE_CPP} ${OPERATORS_CPP}
        include/Affine.h src/core/Affine.cpp
        include/Formula.h src/core/Formula.cpp
        include/Parser.h src/core/Parser.cpp
        include/Computation.h src/core/Computation.cpp
        tests/resnet.cpp src/operators/avgpooling.cpp
        src/operators/batch_flatten.cpp src/operators/matmul.cpp
        tests/layer.h tests/yolo_v1.cpp tests/yolo_v2.cpp
        src/operators/yolo_reorg.cpp src/operators/split.cpp
        tests/yolo_v3.cpp tests/vgg.cpp tests/inception_v3.cpp
        tests/mlp.cpp tests/dqn.cpp tests/densenet.cpp
        tests/synthetic.cpp tests/squeezenet.cpp
        tests/mobilenet.cpp tests/layer.cpp tests/dcgan.cpp
        tests/lstm.cpp tests/resnet_3d.cpp tests/yolo_v4.cpp
        src/operators/pad.cpp tests/yolo_v5.cpp
        src/operators/slice.cpp src/operators/winograd.cpp
        include/Polynomial.h src/core/Polynomial.cpp
        src/core/Producer.cpp include/Producer.h
        src/operators/bilinear.cpp tests/resnext.cpp tests/nasnet.cpp tests/nasnet_a.cpp tests/alexnet.cpp tests/nasrnn.cpp tests/bert.cpp src/operators/transpose.cpp)
