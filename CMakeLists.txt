cmake_minimum_required(VERSION 3.21)
project(NNInteraction)

#set(CUDNN_INCLUDE_PATH /usr/local/cuda/include)
#set(CUDNN_LIBRARY_PATH /usr/local/cuda/lib64)

#include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
#list(APPEND CMAKE_PREFIX_PATH "/Volumes/libs/cpp/libtorch")

set(GCC_COVERAGE_COMPILE_FLAGS "-Wall")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
#set(CMAKE_CUDA_COMPILER nvcc)

find_package(GSL REQUIRED)
find_package(jsoncpp REQUIRED)
#find_package(Torch REQUIRED)

add_executable(NNInteraction main.cpp
        include/qcd/propagators/QuarkPropagator.hpp
        include/qcd/propagators/ScalarDiquarkPropagator.hpp
        include/operators/Projectors.hpp
        include/utils/MatrixInitializers.hpp
        include/utils/dirac/DiracStructures.hpp
        include/utils/dirac/DiracStructuresHelper.hpp
        include/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp
        include/scattering/processes/QuarkExchange.hpp
        include/scattering/ScatteringProcess.hpp
        include/operators/ChargeConjugation.hpp
        include/scattering/basis/TensorBasis.hpp
        include/utils/tensor/Tensor4.hpp
        include/utils/math/Commutator.hpp
        include/numerics/Integration.hpp
        include/scattering/impulse/ExternalImpulseGrid.hpp
        include/scattering/MomentumLoop.hpp
        include/numerics/Integratable.hpp
        include/scattering/impulse/LoopComplexImpulseGrid.hpp
        include/scattering/impulse/LoopImpulseGrid.hpp
        include/numerics/integration/GaussChebyshev.hpp
        include/numerics/integration/GaussLegendre.hpp
        include/numerics/polynomials/LegendrePolynomials.hpp
        include/numerics/roots/NewtonRootFinder.hpp
        src/qcd/propagators/QuarkPropagator.cpp
        src/qcd/propagators/ScalarDiquarkPropagator.cpp
        src/operators/Projectors.cpp
        src/utils/MatrixInitializers.cpp
        src/utils/dirac/DiracStructures.cpp
        src/utils/dirac/DiracStructuresHelper.cpp
        src/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.cpp
        src/scattering/processes/QuarkExchange.cpp
        src/operators/ChargeConjugation.cpp
        src/scattering/basis/TensorBasis.cpp
        src/utils/tensor/Tensor4.cpp
        src/utils/math/Commutator.cpp
        src/numerics/Integration.cpp
        src/scattering/impulse/ExternalImpulseGrid.cpp
        src/scattering/ScatteringProcess.cpp
        src/scattering/MomentumLoop.cpp
        src/scattering/impulse/LoopComplexImpulseGrid.cpp
        src/scattering/impulse/LoopImpulseGrid.cpp
        src/numerics/integration/GaussChebyshev.cpp
        src/numerics/integration/GaussLegendre.cpp
        src/numerics/polynomials/LegendrePolynomials.cpp
        src/numerics/roots/NewtonRootFinder.cpp src/utils/print/PrintGSLElements.cpp include/utils/print/PrintGSLElements.hpp src/scattering/ScatteringProcessHandler.cpp include/scattering/ScatteringProcessHandler.hpp src/scattering/impulse/ZXGrid.cpp include/scattering/impulse/ZXGrid.hpp include/Definitions.h
        src/scattering/processes/DiquarkExchange.cpp
        include/scattering/processes/DiquarkExchange.hpp)

target_link_libraries(NNInteraction GSL::gsl jsoncpp)
#target_link_libraries(NNInteraction "${TORCH_LIBRARIES}")