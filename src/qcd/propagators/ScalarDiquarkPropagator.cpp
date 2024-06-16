//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../../include/qcd/propagators/ScalarDiquarkPropagator.hpp"

double ScalarDiquarkPropagator::D0 = 1.39;
double ScalarDiquarkPropagator::Doo = 0.68;
double ScalarDiquarkPropagator::L2 = 8.0;
double ScalarDiquarkPropagator::D1 = 0.39;

double ScalarDiquarkPropagator::m_sc = 0.8;

gsl_complex ScalarDiquarkPropagator::M(gsl_complex p2)
{
    return gsl_complex_rect(0.8, 0); // GeV
}

void ScalarDiquarkPropagator::D(gsl_vector_complex* p, gsl_complex* diquarkPropScalar)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);
    gsl_complex xSC = gsl_complex_sqrt(p2);     // xSC = sqrt(p2)

    gsl_complex D_SC;
    // D_SC =
    //          D0 / ( 1D0 + xSC ) *
    //          ( 1D0 + D1*xSC + Doo/D0 * xSC**2/L2 ) /
    //              ( 1D0 + xSC/L2 ) *
    //      1 / MM_SC**2
    D_SC = gsl_complex_div_real(gsl_complex_mul(gsl_complex_mul_real(gsl_complex_inverse(gsl_complex_add_real(xSC, 1.0)), D0),
                                                   gsl_complex_div(gsl_complex_add_real( gsl_complex_add(gsl_complex_mul_real(xSC, D1), gsl_complex_mul_real(p2, Doo/(D0 * L2))), 1.0),
                                                                      gsl_complex_add_real(gsl_complex_div_real(xSC, L2), 1.0))),
                                pow(m_sc, 2));

    *diquarkPropScalar = D_SC;
}
