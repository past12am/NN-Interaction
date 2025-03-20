//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include <cassert>

#include "../../../include/gslhacks/GSLComplexOperators.hpp"

#include "../../../include/qcd/propagators/QuarkPropagator.hpp"

#include "../../../include/Definitions.h"
#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/utils/dirac/DiracStructures.hpp"

gsl_complex QuarkPropagator::M(gsl_complex p2)
{
    // 0.008D0 + 0.53D0*EXP(-1.36D0*p2) + 0.12D0*p2*EXP(-1.01D0*p2)
    return gsl_complex_add_real(gsl_complex_add(gsl_complex_mul_real(gsl_complex_exp(gsl_complex_mul_real(p2, -1.36)), 0.53),
                                                   gsl_complex_mul_real(gsl_complex_mul(p2, gsl_complex_exp(gsl_complex_mul_real(p2, -1.01))), 0.12)),
                                0.008); // GeV
}

gsl_complex QuarkPropagator::Z_f(gsl_complex p2, gsl_complex M2)
{
    return sigma_v(p2) * (p2 + M2);
}

gsl_complex QuarkPropagator::sigma_s(gsl_complex M_val, gsl_complex sigma_v_val)
{
    return gsl_complex_mul(sigma_v_val, M_val);
}

gsl_complex QuarkPropagator::sigma_v(gsl_complex p2)
{
    //1.004D0*( p2 + 0.25D0 )/( ( p2 + 0.25D0 )**2 + 0.40D0**2 )   + 1.11D0*EXP(-5.09D0*p2)
    return gsl_complex_add(gsl_complex_div(gsl_complex_mul_real(gsl_complex_add_real(p2, 0.25), 1.004),
                                              gsl_complex_add_real(gsl_complex_pow_real(gsl_complex_add_real(p2, 0.25), 2), pow(0.40, 2))),
                           gsl_complex_mul_real(gsl_complex_exp(gsl_complex_mul_real(p2, -5.09)), 1.11));
}

gsl_complex QuarkPropagator::sigma_v(gsl_complex p2, gsl_complex x_4, double absx, double y, double phi, double X, double Z, bool sign_plus, double eta, double epsilon)
{
    //1.004D0*( p2 + 0.25D0 )/( ( p2 + 0.25D0 )**2 + 0.40D0**2 )   + 1.11D0*EXP(-5.09D0*p2)
    double x2 = absx * absx;

    gsl_complex p2_eps_shifted = calc_p2_with_epsilon_shift(x_4, A_imag(X, eta), D_plusminus(x2, absx, y, phi, X, Z, sign_plus), epsilon);
    return 1.004 * (p2 + 0.25) / (gsl_complex_pow_real(p2_eps_shifted + 0.25, 2) + pow(0.4, 2) + 1.11 * gsl_complex_exp(-5.09 * p2));
}

void QuarkPropagator::S(gsl_vector_complex* p, gsl_matrix_complex* quarkProp)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    gsl_complex sigma_v_val = sigma_v(p2);
    gsl_complex sigma_s_val = sigma_s(M(p2), sigma_v_val);


    // Identity part
    gsl_matrix_complex_set_identity(quarkProp);
    gsl_matrix_complex_scale(quarkProp, sigma_s_val);


    // pSlash = -i pSlash
    DiracStructuresHelper::diracStructures.slash(p, pSlashCurrent);
    gsl_matrix_complex_scale(pSlashCurrent, gsl_complex_rect(0, -1.0));

    gsl_matrix_complex_scale(pSlashCurrent, sigma_v_val);


    // Combine both tensors
    gsl_matrix_complex_add(quarkProp, pSlashCurrent);
}

void QuarkPropagator::S(gsl_vector_complex* p, gsl_matrix_complex* quarkProp, gsl_complex x_4, double absx, double y, double phi, bool sign_plus, double X, double Z,
    double eta, double epsilon)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    gsl_complex sigma_v_val = sigma_v(p2, x_4, absx, y, phi, X, Z, sign_plus, eta, epsilon);
    gsl_complex sigma_s_val = sigma_s(M(p2), sigma_v_val);


    // Identity part
    gsl_matrix_complex_set_identity(quarkProp);
    gsl_matrix_complex_scale(quarkProp, sigma_s_val);


    // pSlash = -i pSlash
    DiracStructuresHelper::diracStructures.slash(p, pSlashCurrent);
    gsl_matrix_complex_scale(pSlashCurrent, gsl_complex_rect(0, -1.0));

    gsl_matrix_complex_scale(pSlashCurrent, sigma_v_val);


    // Combine both tensors
    gsl_matrix_complex_add(quarkProp, pSlashCurrent);
}

QuarkPropagator::~QuarkPropagator()
{
    gsl_matrix_complex_free(pSlashCurrent);
}

QuarkPropagator::QuarkPropagator()
{
    pSlashCurrent = gsl_matrix_complex_alloc(4, 4);
}

gsl_complex QuarkPropagator::calc_p2_with_epsilon_shift(gsl_complex x_4, double A_imag, double D_plusminus,
    double epsilon)
{
    gsl_complex A = gsl_complex_rect(0, A_imag);
    return M_nucleon * M_nucleon * (x_4 * x_4 - 2.0 * (A + epsilon) * x_4 + D_plusminus + gsl_complex_pow_real(A + epsilon, 2));
}

double QuarkPropagator::A_imag(double X, double eta)
{
    return -eta * sqrt(1.0 + X);
}

double QuarkPropagator::D_plusminus(double x2, double absx, double y, double phi, double X, double Z, bool sign_plus)
{
    return x2 + (sign_plus ? 1.0 : -1.0) * sqrt(2.0 * X * (1.0 - Z)) * absx * Omega(y, phi, Z) + X/2.0 * (2.0 - Z);
}

double QuarkPropagator::Omega(double y, double phi, double Z)
{
    // TODO check the calculation for both Quark and Diquark propagator in mathematica
    return y * sqrt((1.0 - Z) / 2.0) - sqrt(1.0 - y*y) * sqrt((1.0 + Z) / 2) * cos(phi);
}
