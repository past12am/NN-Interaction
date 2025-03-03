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
    // valPSquared = P^2
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);

    assert(GSL_IMAG(valPSquared) == 0);
    assert(GSL_REAL(valPSquared) < 0);

    gsl_complex invValPSquared = gsl_complex_inverse(valPSquared);
    assert(GSL_IMAG(invValPSquared) == 0);
    assert(GSL_REAL(invValPSquared) < 0);

    gsl_matrix_complex_memcpy(transvProj, unitM);

    // A <- A + alpha X @ Y^T
    gsl_blas_zgeru(gsl_complex_negative(invValPSquared), P, P, transvProj);

    assert(checkProjectorProperties(transvProj));
}

void Projectors::longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj)
{
    // TODO check Projector properties

    // valPSquared = |P|^2
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);
    assert(GSL_IMAG(valPSquared) == 0);
    assert(GSL_REAL(valPSquared) < 0);

    gsl_matrix_complex_set_zero(longitudProj);
    gsl_blas_zgeru(gsl_complex_inverse(valPSquared), P, P, longitudProj);

    assert(checkProjectorProperties(longitudProj));
}

void Projectors::posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj)
{
    // Find norm of P (for nomalization)
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);
    assert(GSL_IMAG(valPSquared) < 1E-15);
    assert(GSL_REAL(valPSquared) < 0);

    gsl_complex valP = gsl_complex_sqrt(valPSquared);    // - sign is because valPSquared < 0


    // posEnergyProj = slash(P)
    DiracStructuresHelper::diracStructures.slash(P, posEnergyProj);

    // posEnergyProj = slash(P)/valP
    gsl_matrix_complex_scale(posEnergyProj, gsl_complex_div(GSL_COMPLEX_ONE, valP));

    // posEnergyProj = (posEnergyProj + unitM)/2
    gsl_matrix_complex_add(posEnergyProj, unitM);
    gsl_matrix_complex_scale(posEnergyProj, gsl_complex_rect(0.5, 0));

    // posEnergyProj = 1/2 * (unitM + slash(P)/valP)
    assert(checkProjectorProperties(posEnergyProj));
}

bool Projectors::checkProjectorProperties(gsl_matrix_complex* projector)
{
    bool valid = true;
    // projector * projector = projector

    gsl_matrix_complex* tmp = gsl_matrix_complex_alloc(4, 4);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, projector, projector, GSL_COMPLEX_ZERO, tmp);

    for (int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            if(gsl_complex_abs(gsl_complex_sub(gsl_matrix_complex_get(projector, i, j), gsl_matrix_complex_get(tmp, i, j))) > 1E-13)
            {
                valid = false;
            }
        }
    }

    gsl_matrix_complex_free(tmp);

    return valid;
}

const gsl_matrix_complex* Projectors::getUnitM()
{
    return unitM;
}