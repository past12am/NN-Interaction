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
    // TODO check transverse projector implementation

    // TODO: for safety use -|P^2|      (however in place solution works)

    // valPSquared = P^2
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);

    // Note: we use the "-" sign in valPSquqred as the minus in T = 1 - P_mu P_nu
    assert(GSL_IMAG(valPSquared) == 0);
    assert(GSL_REAL(valPSquared) < 0);

    gsl_matrix_complex_memcpy(transvProj, unitM);

    // A <- A + alpha X @ Y^T
    gsl_blas_zgeru(gsl_complex_inverse(valPSquared), P, P, transvProj);
}

void Projectors::longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj)
{
    // TODO check longitudinal projector implementation

    // valPSquared = |P|^2
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);

    GSL_REAL(valPSquared) = abs(GSL_REAL(valPSquared));
    assert(GSL_IMAG(valPSquared) == 0);

    // longProj^(mu,nu) = 1/|p|^2 p^mu p^nu
    // TODO check might need abs(valPSquared)   --> Done, but not beautiful
    const gsl_matrix_complex_view PMatView = gsl_matrix_complex_view_vector(P, 4, 1);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1.0/GSL_REAL(valPSquared), 0.0), &PMatView.matrix, &PMatView.matrix, gsl_complex_rect(0.0, 0.0), longitudProj);
}

void Projectors::posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj)
{
    // TODO check posEnergyProjector implementation

    // Find norm of P (for nomalization)
    gsl_complex valPSquared;
    gsl_blas_zdotu(P, P, &valPSquared);
    double valP = sqrt(gsl_complex_abs(valPSquared));


    // posEnergyProj = slash(P)
    DiracStructuresHelper::diracStructures.slash(P, posEnergyProj);

    // TODO check might need abs(valPSquared)   --> Done
    // posEnergyProj = slash(P)/valP
    gsl_matrix_complex_scale(posEnergyProj, gsl_complex_rect(1.0/valP, 0));

    // posEnergyProj = (posEnergyProj + unitM)/2
    gsl_matrix_complex_add(posEnergyProj, unitM);
    gsl_matrix_complex_scale(posEnergyProj, gsl_complex_rect(0.5, 0));

    // posEnergyProj = 1/2 * (unitM + slash(P)/valP)
}

const gsl_matrix_complex* Projectors::getUnitM()
{
    return unitM;
}
