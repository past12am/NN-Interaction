//
// Created by past12am on 8/17/23.
//

#include "../../../include/utils/math/Commutator.hpp"

#include "gsl/gsl_blas.h"
#include "gsl/gsl_complex_math.h"

gsl_matrix_complex* Commutator::commutator(const gsl_matrix_complex* A, const gsl_matrix_complex* B, gsl_matrix_complex* res)
{
    // res = BA
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), B, A, gsl_complex_rect(0, 0), res);
    // res = AB - res = AB - BA
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), A, B, gsl_complex_rect(-1, 0), res);

    return res;
}
