#include <iostream>
#include "include/scattering/basis/TensorBasis.hpp"
#include "include/scattering/impulse/ExternalImpulseGrid.hpp"
#include "include/scattering/ScatteringProcess.hpp"
#include "include/scattering/processes/QuarkExchange.hpp"

#include <complex>
#include <gsl/gsl_math.h>


int main(int argc, char *argv[])
{
    double m_q = 0.5; // GeV
    double m_d = 0.8; // GeV (scalar diquark)
    gsl_complex M_nucleon = gsl_complex_rect(0, 0.94); // GeV

    double impulse_ir_cutoff = 1E-2;
    double impulse_uv_cutoff = 3E3;
    double impulse_mid_cutoff = 600;

    double tau_upper = 1.0/16.0;
    double tau_lower = 1E-3;

    double z_lower = -1 + 1E-4;
    double z_upper =  1 - 1E-4;

    // Note: grid lengths must be even (edge case not handled)
    QuarkExchange scattering(15, 30, tau_lower, tau_upper, z_lower, z_upper, M_nucleon,
                                        200, 40, 20, 20, gsl_complex_rect(0.19, 0));
    scattering.integrate(1E3);
    scattering.buildScatteringMatrix(M_nucleon);


    scattering.store_scattering_amplitude(argv[1]);

    return 0;
}
