//
// Created by past12am on 8/2/23.
//

#include "../../include/utils/DiracStructures.hpp"

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>




DiracStructures::DiracStructures()
{
    for(int i = 0; i < 4; i++)
    {
        gamma[i] = gsl_matrix_complex_alloc(4, 4);
        gsl_matrix_complex_set_zero(gamma[i]);
    }

    // gamma_1 (= gamma[0])
    gsl_matrix_complex_set(gamma[0], 0, 3, gsl_complex_rect(0, -1));
    gsl_matrix_complex_set(gamma[0], 1, 2, gsl_complex_rect(0, -1));
    gsl_matrix_complex_set(gamma[0], 2, 1, gsl_complex_rect(0, 1));
    gsl_matrix_complex_set(gamma[0], 3, 0, gsl_complex_rect(0, 1));

    // gamma_2 (= gamma[1])
    gsl_matrix_complex_set(gamma[1], 0, 3, gsl_complex_rect(-1, 0));
    gsl_matrix_complex_set(gamma[1], 1, 2, gsl_complex_rect(1, 0));
    gsl_matrix_complex_set(gamma[1], 2, 1, gsl_complex_rect(1, 0));
    gsl_matrix_complex_set(gamma[1], 3, 0, gsl_complex_rect(-1, 0));

    // gamma_3 (= gamma[2])
    gsl_matrix_complex_set(gamma[2], 0, 2, gsl_complex_rect(0, -1));
    gsl_matrix_complex_set(gamma[2], 1, 3, gsl_complex_rect(0, 1));
    gsl_matrix_complex_set(gamma[2], 2, 0, gsl_complex_rect(0, 1));
    gsl_matrix_complex_set(gamma[2], 3, 1, gsl_complex_rect(0, -1));

    // gamma_4 (= gamma[3])
    gsl_matrix_complex_set(gamma[3], 0, 0, gsl_complex_rect(1, 0));
    gsl_matrix_complex_set(gamma[3], 1, 1, gsl_complex_rect(1, 0));
    gsl_matrix_complex_set(gamma[3], 2, 2, gsl_complex_rect(-1, 0));
    gsl_matrix_complex_set(gamma[3], 3, 3, gsl_complex_rect(-1, 0));


    gamma5 = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp1 = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp2 = gsl_matrix_complex_alloc(4, 4);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), gamma[0], gamma[1], gsl_complex_rect(0, 0), tmp1);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), gamma[2], gamma[3], gsl_complex_rect(0, 0), tmp2);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), tmp1, tmp2, gsl_complex_rect(0, 0), gamma5);
    gsl_matrix_complex_free(tmp2);
    gsl_matrix_complex_free(tmp1);
}

void DiracStructures::slash(gsl_vector_complex* p, gsl_matrix_complex* pSlash)
{
    gsl_matrix_complex_set_zero(pSlash);

    gsl_matrix_complex* tempMat = gsl_matrix_complex_alloc(4, 4);
    for(int i = 0; i < 4; i++)
    {
        gsl_matrix_complex_memcpy(tempMat, gamma[i]);
        gsl_matrix_complex_scale(tempMat, gsl_vector_complex_get(p, i));

        gsl_matrix_complex_add(pSlash, tempMat);
    }
    gsl_matrix_complex_free(tempMat);
}
