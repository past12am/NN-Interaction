//
// Created by past12am on 8/2/23.
//

#include "../../include/operators/Projectors.hpp"
#include "../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../include/utils/MatrixInitializers.hpp"

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <cassert>


const gsl_matrix_complex* Projectors::unitM = MatrixInitializers::generateUnitM();

void Projectors::transverseProjector(gsl_vector_complex* P, gsl_matrix_complex* transvProj)
{
    // valPSquared = |P|^2
    gsl_complex valPSquared;
    gsl_blas_zdotc(P, P, &valPSquared);

    assert(GSL_IMAG(valPSquared) == 0);

    // transvProj^(mu,nu) = -1/|p|^2 p^mu p^nu
    const gsl_matrix_complex_view PMatView = gsl_matrix_complex_view_vector(P, 4, 1);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(-1.0/GSL_REAL(valPSquared), 0.0), &PMatView.matrix, &PMatView.matrix, gsl_complex_rect(0.0, 0.0), transvProj);

    // transvProj = -1/|p|^2 p^mu p^nu + kronD
    gsl_matrix_complex_add(transvProj, unitM);
}

void Projectors::longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj)
{
    // valPSquared = |P|^2
    gsl_complex valPSquared;
    gsl_blas_zdotc(P, P, &valPSquared);

    assert(GSL_IMAG(valPSquared) == 0);

    // longProj^(mu,nu) = 1/|p|^2 p^mu p^nu
    const gsl_matrix_complex_view PMatView = gsl_matrix_complex_view_vector(P, 4, 1);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1.0/GSL_REAL(valPSquared), 0.0), &PMatView.matrix, &PMatView.matrix, gsl_complex_rect(0.0, 0.0), longitudProj);
}

void Projectors::posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj)
{
    // Find norm of P
    gsl_complex valPSquared;
    gsl_blas_zdotc(P, P, &valPSquared);
    gsl_complex valP = gsl_complex_sqrt(valPSquared);
    assert(GSL_IMAG(valP) == 0);
    assert(GSL_REAL(valP) > 0);


    // posEnergyProj = slash(P)
    DiracStructuresHelper::diracStructures.slash(P, posEnergyProj);

    // posEnergyProj = slash(P)/valP
    gsl_matrix_complex_scale(posEnergyProj, valP);

    // posEnergyProj = (posEnergyProj + unitM)/2
    gsl_matrix_complex_add(posEnergyProj, unitM);
    gsl_matrix_complex_scale(posEnergyProj, gsl_complex_rect(0.5, 0));

    // posEnergyProj = (unitM + slash(P)/valP)/2
}

const gsl_matrix_complex* Projectors::getUnitM()
{
    return unitM;
}
