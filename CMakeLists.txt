cmake_minimum_required(VERSION 3.21)
project(NNInteraction)

find_package(GSL REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(NNInteraction main.cpp src/propagators/QuarkPropagator.cpp include/propagators/QuarkPropagator.hpp src/propagators/ScalarDiquarkPropagator.cpp include/propagators/ScalarDiquarkPropagator.hpp src/utils/ProjectorsHelper.cpp include/utils/ProjectorsHelper.hpp src/utils/MatrixInitializers.cpp include/utils/MatrixInitializers.hpp src/utils/DiracStructures.cpp include/utils/DiracStructures.hpp src/utils/DiracStructuresHelper.cpp include/utils/DiracStructuresHelper.hpp src/amplitudes/ScalarQuakDiquarkAmplitude.cpp include/amplitudes/ScalarQuakDiquarkAmplitude.hpp)
target_link_libraries(NNInteraction GSL::gsl)