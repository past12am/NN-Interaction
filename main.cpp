#include <iostream>
#include "include/Definitions.h"
#include "include/scattering/basis/TensorBasis.hpp"
#include "include/scattering/impulse/ExternalImpulseGrid.hpp"
#include "include/scattering/ScatteringProcess.hpp"
#include "include/scattering/processes/QuarkExchange.hpp"
#include "include/scattering/ScatteringProcessHandler.hpp"

#include <complex>
#include <gsl/gsl_math.h>


int main(int argc, char *argv[])
{
    double m_q = 0.5; // GeV
    double m_d = 0.8; // GeV (scalar diquark)
    gsl_complex M_nucleon = gsl_complex_rect(0.94, 0); // GeV

    double eta = 0.25;

    double impulse_ir_cutoff = 1E-2;
    double impulse_uv_cutoff = 3E3;
    double impulse_mid_cutoff = 600;

    double tau_upper = 0.5;
    double tau_lower = 1E-3;

    double z_lower = -1 + 1E-4;
    double z_upper =  1 - 1E-4;

    double a_imag_lower = 0.7;
    double a_imag_upper = 1.3;

    // Note: grid lengths must be even (edge case not handled)
    /*
    QuarkExchange scattering(15, 30, tau_lower, tau_upper, z_lower, z_upper, M_nucleon,
                                        200, 40, 20, 20, gsl_complex_rect(0.19, 0));
    scattering.integrate(1E3);
    scattering.buildScatteringMatrix();

    scattering.store_scattering_amplitude(argv[1]);
     */

    if(argc < 2)
    {
        std::cout << "Need to specify output path" << std::endl;
    }

    int numThreads = NUM_THREADS;
    int lenTau = 15;
    int lenZ = 20;
    int lenA = 10;
    ScatteringProcessHandler<QuarkExchange> scatteringProcessHandler(numThreads, lenTau, lenZ, lenA, 80, 30, 7, 7,
                                                                     eta, tau_lower, tau_upper,
                                                                     z_lower, z_upper, a_imag_lower, a_imag_upper, M_nucleon);
    scatteringProcessHandler.calculateScattering(1E3);
    scatteringProcessHandler.store_scattering_amplitude(argv[1]);

    return 0;
}
