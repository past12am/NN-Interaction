//
// Created by past12am on 10/14/23.
//

#ifndef NNINTERACTION_DEFINITIONS_H
#define NNINTERACTION_DEFINITIONS_H

#include <ostream>

#define NUM_THREADS 6

# define BASIS Basis::T

# define DIQUARK_TYPE_1 DiquarkType::scalar
# define DIQUARK_TYPE_2 DiquarkType::scalar

# define AMPLITUDE_ISOSPIN 0

# define QUARK_EXCHANGE_DATA_DIRNAME = "quark_exchange/"
# define DIQUARK_EXCHANGE_DATA_DIRNAME = "diquark_exchange/"

enum class DiquarkType
{
    scalar = 0,
    axialvector = 1
};

inline std::ostream& operator<<(std::ostream& os, DiquarkType dqtype)
{
    switch (dqtype) {
        case DiquarkType::scalar        :  os << "scalar";
            break;
        case DiquarkType::axialvector   : os << "axialvector";
            break;
        default         : os.setstate(std::ios_base::failbit);
    }
    return os;
}


enum class Basis
{
    tau = 0,
    T = 1
};

inline std::ostream& operator<<(std::ostream& os, Basis basis)
{
    switch (basis) {
        case Basis::tau : os << "tau";
            break;
        case Basis::T   : os << "T";
            break;
        default         : os.setstate(std::ios_base::failbit);
    }
    return os;
}

#endif //NNINTERACTION_DEFINITIONS_H
