//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include <cassert>
#include "../../../include/qcd/propagators/QuarkPropagator.hpp"
#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/utils/dirac/DiracStructures.hpp"

gsl_complex QuarkPropagator::M(gsl_complex p2)
{
    // 0.008D0 + 0.53D0*EXP(-1.36D0*p2) + 0.12D0*p2*EXP(-1.01D0*p2)
    return gsl_complex_add_real(gsl_complex_add(gsl_complex_mul_real(gsl_complex_exp(gsl_complex_mul_real(p2, -1.36)), 0.53),
                                                   gsl_complex_mul_real(gsl_complex_mul(p2, gsl_complex_exp(gsl_complex_mul_real(p2, -1.01))), 0.12)),
                                0.008); // GeV
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

QuarkPropagator::~QuarkPropagator()
{
    gsl_matrix_complex_free(pSlashCurrent);
}

QuarkPropagator::QuarkPropagator()
{
    pSlashCurrent = gsl_matrix_complex_alloc(4, 4);
}

