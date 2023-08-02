//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../include/amplitudes/ScalarQuakDiquarkAmplitude.hpp"

ScalarQuakDiquarkAmplitude::ScalarQuakDiquarkAmplitude()
{
    posEnergyProj = gsl_matrix_complex_alloc(4, 4);
}

ScalarQuakDiquarkAmplitude::~ScalarQuakDiquarkAmplitude()
{
    gsl_matrix_complex_free(posEnergyProj);
}

void ScalarQuakDiquarkAmplitude::Phi(gsl_vector_complex* p, gsl_vector_complex* P, gsl_matrix_complex* quarkDiquarkAmp)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    // Only leading tensor --> tau_1 = matE --> consider Lambda+
    projectors.posEnergyProjector(P, posEnergyProj);

    // quarkDiquarkAmp = f(p2) * posEnergyProj(P)
    gsl_matrix_complex_memcpy(quarkDiquarkAmp, posEnergyProj);
    gsl_matrix_complex_scale(quarkDiquarkAmp, f(p2));
}

gsl_complex ScalarQuakDiquarkAmplitude::f(gsl_complex p2)
{
    // f = (c1 + c2*p^2) * e^(-c3*p^2)
    return gsl_complex_mul(gsl_complex_add(c1, gsl_complex_mul(c2, p2)), gsl_complex_exp(gsl_complex_mul(gsl_complex_negative(c3), p2)));
}
