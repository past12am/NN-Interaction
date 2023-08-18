//
// Created by past12am on 8/17/23.
//

#ifndef NNINTERACTION_COMMUTATOR_HPP
#define NNINTERACTION_COMMUTATOR_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

class Commutator
{
    public:
        static gsl_matrix_complex* commutator(const gsl_matrix_complex* A, const gsl_matrix_complex* B, gsl_matrix_complex* res);
};


#endif //NNINTERACTION_COMMUTATOR_HPP
