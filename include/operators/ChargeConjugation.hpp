//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_CHARGECONJUGATION_HPP
#define NNINTERACTION_CHARGECONJUGATION_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

class ChargeConjugation
{
    public:
        static const gsl_matrix_complex* chargeConjMatrix;

        static void chargeConj(gsl_matrix_complex* mat);
};


#endif //NNINTERACTION_CHARGECONJUGATION_HPP
