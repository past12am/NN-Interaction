cmake_minimum_required(VERSION 3.21)
project(NNInteraction)

set(CUDNN_INCLUDE_PATH /usr/local/cuda/include)
set(CUDNN_LIBRARY_PATH /usr/local/cuda/lib64)

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
list(APPEND CMAKE_PREFIX_PATH "/Volumes/libs/cpp/libtorch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_COMPILER nvcc)

find_package(GSL REQUIRED)
find_package(Torch REQUIRED)

add_executable(NNInteraction main.cpp src/propagators/QuarkPropagator.cpp include/propagators/QuarkPropagator.hpp src/propagators/ScalarDiquarkPropagator.cpp include/propagators/ScalarDiquarkPropagator.hpp src/operators/ProjectorsHelper.cpp include/operators/ProjectorsHelper.hpp src/utils/MatrixInitializers.cpp include/utils/MatrixInitializers.hpp src/utils/DiracStructures.cpp include/utils/DiracStructures.hpp src/utils/DiracStructuresHelper.cpp include/utils/DiracStructuresHelper.hpp src/amplitudes/ScalarQuarkDiquarkAmplitude.cpp include/amplitudes/ScalarQuarkDiquarkAmplitude.hpp src/scattering/processes/QuarkExchange.cpp include/scattering/processes/QuarkExchange.hpp include/scattering/ScatteringProcess.hpp src/operators/ChargeConjugation.cpp include/operators/ChargeConjugation.hpp src/scattering/basis/TensorBasis.cpp include/scattering/basis/TensorBasis.hpp src/utils/Tensor4.cpp include/utils/Tensor4.hpp src/utils/Commutator.cpp include/utils/Commutator.hpp src/numerics/Integration.cpp include/numerics/Integration.hpp)
target_link_libraries(NNInteraction GSL::gsl)
#target_link_libraries(NNInteraction "${TORCH_LIBRARIES}")