//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

#include "../../../include/gslhacks/GSLComplexOperators.hpp"

#include "../../../include/qcd/propagators/ScalarDiquarkPropagator.hpp"

#include "../../../include/Definitions.h"

double ScalarDiquarkPropagator::D0 = 1.39;
double ScalarDiquarkPropagator::Doo = 0.68;
double ScalarDiquarkPropagator::L2 = 8.0;
double ScalarDiquarkPropagator::D1 = 0.39;

double ScalarDiquarkPropagator::m_sc = 0.8;

void ScalarDiquarkPropagator::D(gsl_vector_complex* p, gsl_complex* diquarkPropScalar)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);
    gsl_complex xSC = gsl_complex_div_real(p2, m_sc * m_sc);     // xSC = p^2 / M_dq^2

    gsl_complex D_SC;
    // D_SC = D0 / ( 1D0 + xSC ) * ( 1D0 + D1*xSC + Doo/D0 * xSC**2/L2 ) / ( 1D0 + xSC/L2 ) / MM_SC**2
    D_SC = D0 / (1.0 + xSC) * (1.0 + D1 * xSC + Doo/D0 * (xSC * xSC)/L2) / (1.0 + xSC/L2) / (m_sc * m_sc);

    *diquarkPropScalar = D_SC;
}

void ScalarDiquarkPropagator::D(gsl_complex x_4, double absx, double y, double phi, bool sign_plus, double X, double Z, double eta, gsl_complex* diquarkPropScalar, double epsilon)
{
    double x2 = absx * absx;
    double m_sc2 = (m_sc * m_sc);

    gsl_complex p2 = M_nucleon * (gsl_complex_pow_real(x_4, 2) + x2);

    gsl_complex p2_pole_shifted = calc_p2_with_epsilon_shift(x_4, A_imag(X, eta), D_plusminus(x2, absx, y, phi, X, Z, sign_plus), epsilon);

    gsl_complex xSC = gsl_complex_div_real(p2, m_sc * m_sc);     // xSC = p^2 / M_dq^2

    // seperate Poles
    // 1st pole    1 / ( 1D0 + xSC ) = M_dq^2 / (M_dq^2 + p^2)
    gsl_complex pole1 = p2_pole_shifted + m_sc2;

    // 2nd pole    1 / ( 1D0 + xSC/L2 ) = L2 / (L2 + xSC) = M_dq^2 L2 / (L2 * M_dq^2 + p^2)        // TODO check calculation on paper
    gsl_complex pole2 = p2_pole_shifted + L2 * m_sc2;




    // TODO check
    // D_SC = D0 / ( 1D0 + xSC ) * ( 1D0 + D1*xSC + Doo/D0 * xSC**2/L2 ) / ( 1D0 + xSC/L2 ) / MM_SC**2
    //      = m_sc2 D0 / pole1 * ( 1D0 + D1*xSC + Doo/D0 * xSC**2/L2 ) * m_sc2 L2 / pole2 / m_sc2
    //      = m_sc2 D0 / pole1 * ( 1D0 + D1*xSC + Doo/D0 * xSC**2/L2 ) * L2 / pole2
    gsl_complex D_SC = m_sc2 * D0 / pole1 * (1.0 + D1 * xSC + Doo/D0 * (xSC * xSC)/L2) * L2 / pole2 ;

    // TODO (what is written here is only to suppress Wunused-but-set warning
    *diquarkPropScalar = D_SC;
}

gsl_complex ScalarDiquarkPropagator::calc_p2_with_epsilon_shift(gsl_complex x_4, double A_imag, double D_plusminus, double epsilon)
{
    gsl_complex A = gsl_complex_rect(0, A_imag);
    return x_4 * x_4 - 2.0 * (A - epsilon) * x_4 + D_plusminus + gsl_complex_pow_real(A - epsilon, 2);
}


double ScalarDiquarkPropagator::A_imag(double X, double eta)
{
    return (1.0 - eta) * sqrt(1.0 + X);
}

double ScalarDiquarkPropagator::D_plusminus(double x2, double absx, double y, double phi, double X, double Z, bool sign_plus)
{
    return x2 + (sign_plus ? 1.0 : -1.0) * sqrt(2.0 * X * (1.0 + Z)) * absx * Omega(y, phi, Z) + X/2.0 * (1.0 + Z);
}

double ScalarDiquarkPropagator::Omega(double y, double phi, double Z)
{
    return y * sqrt((1.0 + Z)/2.0) + sqrt(1.0 - pow(y, 2)) * sqrt((1.0 - Z)/2.0) * cos(phi);
}
