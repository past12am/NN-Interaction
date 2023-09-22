//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../../include/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp"

ScalarQuarkDiquarkAmplitude::ScalarQuarkDiquarkAmplitude()
{
    posEnergyProj = gsl_matrix_complex_alloc(4, 4);
}

ScalarQuarkDiquarkAmplitude::~ScalarQuarkDiquarkAmplitude()
{
    gsl_matrix_complex_free(posEnergyProj);
}

void ScalarQuarkDiquarkAmplitude::Phi(gsl_vector_complex* p, gsl_vector_complex* P, gsl_matrix_complex* quarkDiquarkAmp)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    // Only leading tensor --> tau_1 = matE --> consider Lambda+
    Projectors::posEnergyProjector(P, posEnergyProj);

    // quarkDiquarkAmp = f(p2) * posEnergyProj(P)
    gsl_matrix_complex_memcpy(quarkDiquarkAmp, posEnergyProj);
    gsl_matrix_complex_scale(quarkDiquarkAmp, f(p2));
}

gsl_complex ScalarQuarkDiquarkAmplitude::f(gsl_complex p2)
{
    // f = (c1 + c2*p^2) * e^(-c3*p^2)
    gsl_complex term1 = gsl_complex_add(c1, gsl_complex_mul(c2, p2));
    gsl_complex exponential_param = gsl_complex_mul(gsl_complex_negative(c3), p2);
    gsl_complex term2 = gsl_complex_exp(exponential_param);

    return gsl_complex_mul(term1, term2);
}
