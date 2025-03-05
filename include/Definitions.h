//
// Created by past12am on 10/14/23.
//

#ifndef NNINTERACTION_DEFINITIONS_H
#define NNINTERACTION_DEFINITIONS_H

#include <ostream>

#define NUM_THREADS 12

# define BASIS Basis::tau
# define PROJECTION_BASIS Basis::tau_prime

# define DIQUARK_TYPE_1 DiquarkType::SCALAR
# define DIQUARK_TYPE_2 DiquarkType::SCALAR

# define AMPLITUDE_ISOSPIN 0

#define SCATTERING_PROCESS_TYPE ScatteringProcessType::QUARK_EXCHANGE

#define INVERT_STRATEGY InvertStrategy::ANALYTIC

#define M_nucleon 0.94

enum class InvertStrategy
{
        ANALYTIC,
        NUMERIC_MATRIX_INVERSE
};

inline std::ostream& operator<<(std::ostream& os, InvertStrategy invertStrategy)
{
    switch (invertStrategy) {
        case InvertStrategy::ANALYTIC               :  os << "analytic";
            break;
        case InvertStrategy::NUMERIC_MATRIX_INVERSE : os << "numeric_matrix_inverse";
            break;
        default: os.setstate(std::ios_base::failbit);
    }
    return os;
}

enum class ScatteringProcessType
{
        QUARK_EXCHANGE,
        DIQUARK_EXCHANGE
};

inline std::ostream& operator<<(std::ostream& os, ScatteringProcessType scatteringType)
{
    switch (scatteringType) {
        case ScatteringProcessType::QUARK_EXCHANGE      :  os << "quark_exchange";
            break;
        case ScatteringProcessType::DIQUARK_EXCHANGE    : os << "diquark_exchange";
            break;
        default         : os.setstate(std::ios_base::failbit);
    }
    return os;
}


enum class DiquarkType
{
    SCALAR = 0,
    AXIALVECTOR = 1
};

inline std::ostream& operator<<(std::ostream& os, DiquarkType dqtype)
{
    switch (dqtype) {
        case DiquarkType::SCALAR        :  os << "scalar";
            break;
        case DiquarkType::AXIALVECTOR   : os << "axialvector";
            break;
        default         : os.setstate(std::ios_base::failbit);
    }
    return os;
}


enum class Basis
{
    tau = 0,
    T = 1,
    tau_prime = 2
};

inline std::ostream& operator<<(std::ostream& os, Basis basis)
{
    switch (basis) {
        case Basis::tau         : os << "tau";
            break;
        case Basis::T           : os << "T";
            break;
        case Basis::tau_prime   : os << "tau_prime";
            break;
        default         : os.setstate(std::ios_base::failbit);
    }
    return os;
}

#endif //NNINTERACTION_DEFINITIONS_H
