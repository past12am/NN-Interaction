//
// Created by past12am on 8/3/23.
//

#include "../../include/Definitions.h"
#include "../../include/operators/ChargeConjugation.hpp"

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

static gsl_matrix_complex** generateTmpMatrixArray()
{
    gsl_matrix_complex** tmpArray = new gsl_matrix_complex*[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        tmpArray[i] = gsl_matrix_complex_alloc(4, 4);
    }

    return tmpArray;
}

gsl_matrix_complex** ChargeConjugation::tmpArray = generateTmpMatrixArray();

std::mutex ChargeConjugation::tmpMutexArray[NUM_THREADS] = {};

static const gsl_matrix_complex* generateChargeConjMatrix()
{
    gsl_matrix_complex* chargeConjMat = gsl_matrix_complex_alloc(4, 4);

    // C = gam_4 gam_2
    gsl_matrix_complex_set_zero(chargeConjMat);
    gsl_matrix_complex_set(chargeConjMat, 0, 3, gsl_complex_rect(-1, 0));
    gsl_matrix_complex_set(chargeConjMat, 1, 2, gsl_complex_rect(1, 0));
    gsl_matrix_complex_set(chargeConjMat, 2, 1, gsl_complex_rect(-1, 0));
    gsl_matrix_complex_set(chargeConjMat, 3, 0, gsl_complex_rect(1, 0));

    return chargeConjMat;
}

const gsl_matrix_complex* ChargeConjugation::chargeConjMatrix = generateChargeConjMatrix();


void ChargeConjugation::chargeConj(gsl_matrix_complex* mat)
{
    chargeConj(mat, 0);
}

void ChargeConjugation::chargeConj(gsl_matrix_complex* mat, int thread_number)
{
    tmpMutexArray[thread_number].lock();
    chargeConj(mat, ChargeConjugation::tmpArray[thread_number]);
    tmpMutexArray[thread_number].unlock();
}

void ChargeConjugation::chargeConj(gsl_matrix_complex* mat, gsl_matrix_complex* tmp)
{
    gsl_blas_zgemm(CblasNoTrans, CblasTrans, gsl_complex_rect(1, 0), chargeConjMatrix, mat, gsl_complex_rect(0, 0), tmp);
    gsl_blas_zgemm(CblasNoTrans, CblasTrans, gsl_complex_rect(1, 0), tmp, chargeConjMatrix, gsl_complex_rect(0, 0), mat);
}
