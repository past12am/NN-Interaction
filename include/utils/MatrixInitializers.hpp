//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_MATRIXINITIALIZERS_HPP
#define NNINTERACTION_MATRIXINITIALIZERS_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

class MatrixInitializers
{
    public:
        static const gsl_matrix_complex* generateUnitM();
};


#endif //NNINTERACTION_MATRIXINITIALIZERS_HPP
