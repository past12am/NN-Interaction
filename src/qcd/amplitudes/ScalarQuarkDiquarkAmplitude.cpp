//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../../include/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp"
#include "../../../include/operators/ChargeConjugation.hpp"

ScalarQuarkDiquarkAmplitude::ScalarQuarkDiquarkAmplitude()
{
    posEnergyProj = gsl_matrix_complex_alloc(4, 4);

    p_copy = gsl_vector_complex_alloc(4);
    P_copy = gsl_vector_complex_alloc(4);
}

ScalarQuarkDiquarkAmplitude::~ScalarQuarkDiquarkAmplitude()
{
    gsl_vector_complex_free(P_copy);
    gsl_vector_complex_free(p_copy);

    gsl_matrix_complex_free(posEnergyProj);
}

void ScalarQuarkDiquarkAmplitude::Phi(gsl_vector_complex* p, gsl_vector_complex* P, bool chargeConj, int threadIdx, gsl_matrix_complex* quarkDiquarkAmp)
{
    // Charge Conjugation
    //  ChargeConj(Phi(p, P)) = C Phi(-p, -P)^T C^T

    gsl_vector_complex_memcpy(p_copy, p);
    gsl_vector_complex_memcpy(P_copy, P);

    if(chargeConj)
    {
        gsl_vector_complex_scale(p_copy, gsl_complex_rect(-1, 0));
        gsl_vector_complex_scale(P_copy, gsl_complex_rect(-1, 0));
    }

    gsl_complex p2;
    gsl_blas_zdotu(p_copy, p_copy, &p2);

    // Only leading tensor --> tau_1 = matE --> consider Lambda+
    Projectors::posEnergyProjector(P_copy, posEnergyProj);

    // quarkDiquarkAmp = f(p2) * posEnergyProj(P)
    gsl_matrix_complex_memcpy(quarkDiquarkAmp, posEnergyProj);
    gsl_matrix_complex_scale(quarkDiquarkAmp, f(p2));


    if(chargeConj)
    {
        ChargeConjugation::chargeConj(quarkDiquarkAmp, threadIdx);
    }
}

gsl_complex ScalarQuarkDiquarkAmplitude::f(gsl_complex p2)
{
    // f = (c1 + c2*p^2) * e^(-c3*p^2)
    gsl_complex term1 = gsl_complex_add(c1, gsl_complex_mul(c2, p2));
    gsl_complex exponential_param = gsl_complex_mul(gsl_complex_negative(c3), p2);
    gsl_complex term2 = gsl_complex_exp(exponential_param);

    return gsl_complex_mul(term1, term2);
}
