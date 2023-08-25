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

add_executable(NNInteraction main.cpp src/qcd/propagators/QuarkPropagator.cpp include/qcd/propagators/QuarkPropagator.hpp src/qcd/propagators/ScalarDiquarkPropagator.cpp include/qcd/propagators/ScalarDiquarkPropagator.hpp src/operators/Projectors.cpp include/operators/Projectors.hpp src/utils/MatrixInitializers.cpp include/utils/MatrixInitializers.hpp src/utils/dirac/DiracStructures.cpp include/utils/dirac/DiracStructures.hpp src/utils/dirac/DiracStructuresHelper.cpp include/utils/dirac/DiracStructuresHelper.hpp src/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.cpp include/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp src/scattering/processes/QuarkExchange.cpp include/scattering/processes/QuarkExchange.hpp include/scattering/ScatteringProcess.hpp src/operators/ChargeConjugation.cpp include/operators/ChargeConjugation.hpp src/scattering/basis/TensorBasis.cpp include/scattering/basis/TensorBasis.hpp src/utils/tensor/Tensor4.cpp include/utils/tensor/Tensor4.hpp src/utils/math/Commutator.cpp include/utils/math/Commutator.hpp src/numerics/Integration.cpp include/numerics/Integration.hpp src/scattering/NNInteraction.cpp include/scattering/NNInteraction.hpp src/scattering/impulse/ExternalImpulseGrid.cpp include/scattering/impulse/ExternalImpulseGrid.hpp src/scattering/ScatteringProcess.cpp src/scattering/MomentumLoop.cpp include/scattering/MomentumLoop.hpp include/numerics/Integratable.hpp src/scattering/impulse/LoopComplexImpulseGrid.cpp include/scattering/impulse/LoopComplexImpulseGrid.hpp src/scattering/impulse/LoopImpulseGrid.cpp include/scattering/impulse/LoopImpulseGrid.hpp)
target_link_libraries(NNInteraction GSL::gsl)
#target_link_libraries(NNInteraction "${TORCH_LIBRARIES}")