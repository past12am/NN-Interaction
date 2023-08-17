//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_TENSORBASIS_HPP
#define NNINTERACTION_TENSORBASIS_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

#include "../../utils/Tensor4.hpp"
#include "../../operators/ProjectorsHelper.hpp"


class TensorBasis
{
    private:

        void matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C,  gsl_matrix_complex* tmp, gsl_matrix_complex* res);

    public:
        Tensor4<4, 4, 4, 4> tau[8];

        TensorBasis(gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f, gsl_vector_complex* k_i, gsl_vector_complex* P, gsl_vector_complex* K);

        explicit operator std::string() const;
};


#endif //NNINTERACTION_TENSORBASIS_HPP
