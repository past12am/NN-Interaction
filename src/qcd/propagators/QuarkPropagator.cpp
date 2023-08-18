//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "../../../include/qcd/propagators/QuarkPropagator.hpp"
#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/utils/dirac/DiracStructures.hpp"

gsl_complex QuarkPropagator::M(gsl_complex p2)
{
    return gsl_complex_rect(0.5, 0); // GeV
}

gsl_complex QuarkPropagator::A(gsl_complex p2, gsl_complex mu2)
{
    return gsl_complex_rect(1, 0);
}

void QuarkPropagator::S(gsl_vector_complex* p, gsl_complex mu2, gsl_matrix_complex* quarkProp)
{
    gsl_complex p2;
    gsl_blas_zdotu(p, p, &p2);

    // pref = 1/(A(p2, mu2) * (p2 + M(p2)^2))
    gsl_complex pref = gsl_complex_div(gsl_complex_rect(1.0, 0), gsl_complex_mul(A(p2, mu2), gsl_complex_add(p2, gsl_complex_pow_real(M(p2), 2.0))));

    DiracStructuresHelper::diracStructures.slash(p, pSlashCurrent);

    // pSlash = -i pSlash
    gsl_matrix_complex_scale(pSlashCurrent, gsl_complex_rect(0, -1));

    // quarkProp = M(p2) * unitM
    gsl_matrix_complex_set_identity(quarkProp);
    gsl_matrix_complex_scale(quarkProp, M(p2));

    // quarkProp = pref * (-i pSlash + quarkProp)
    gsl_matrix_complex_add(quarkProp, pSlashCurrent);
    gsl_matrix_complex_scale(quarkProp, pref);

    // quarkProp = 1/(A(p2, mu2) * (p2 + M(p2)^2)) * (-i pSlash + M(p2) * unitM)
}

QuarkPropagator::~QuarkPropagator()
{
    gsl_matrix_complex_free(pSlashCurrent);
}

QuarkPropagator::QuarkPropagator()
{
    pSlashCurrent = gsl_matrix_complex_alloc(4, 4);
}

