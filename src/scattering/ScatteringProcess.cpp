//
// Created by past12am on 8/18/23.
//

#include "../../include/scattering/ScatteringProcess.hpp"


ScatteringProcess::ScatteringProcess(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, gsl_complex M_nucleon) : externalImpulseGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, M_nucleon),
                                                                                                   tensorBasis(&externalImpulseGrid)
{
}

void ScatteringProcess::calc_l(double l2, double z, double y, double phi, gsl_vector_complex* l)
{
    gsl_vector_complex_set(l, 0, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(l, 1, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * y, 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(z, 0));

    gsl_vector_complex_scale(l, gsl_complex_rect(sqrt(l2), 0));
}
