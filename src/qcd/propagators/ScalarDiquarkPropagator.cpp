//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../../include/qcd/propagators/ScalarDiquarkPropagator.hpp"

gsl_complex ScalarDiquarkPropagator::M(gsl_complex p2)
{
    return gsl_complex_rect(0.8, 0); // GeV
}

void ScalarDiquarkPropagator::D(gsl_vector_complex* p, gsl_complex* diquarkPropScalar)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    // diquarkPropScalar = 1/(p2 + M(p2)^2)
    *diquarkPropScalar = gsl_complex_div(gsl_complex_rect(1.0, 0), gsl_complex_add(p2, gsl_complex_pow_real(M(p2), 2.0)));
}
