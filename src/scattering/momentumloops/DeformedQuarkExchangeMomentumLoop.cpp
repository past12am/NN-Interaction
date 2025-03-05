//
// Created by past12am on 3/5/25.
//

#include "../../../include/scattering/momentumloops/DeformedQuarkExchangeMomentumLoop.hpp"

#include <gsl/gsl_complex_math.h>

#include "../../../include/Definitions.h"

void DeformedQuarkExchangeMomentumLoop::calc_k(double k_4, double absk, double y, double phi, gsl_vector_complex* k)
{
    gsl_vector_complex_set(k, 0, gsl_complex_rect(M_nucleon * absk * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(k, 0, gsl_complex_rect(M_nucleon * absk * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(k, 0, gsl_complex_rect(M_nucleon * absk * y, 0));
    gsl_vector_complex_set(k, 0, gsl_complex_rect(M_nucleon * k_4, 0));
}
