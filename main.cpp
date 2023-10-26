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

    double eta = 0.5;

    double impulse_ir_cutoff = 1E-2;
    double impulse_uv_cutoff = 3E3;
    double impulse_mid_cutoff = 600;

    double X_upper = 1;
    double X_lower = 1E-3;

    double z_lower = -1 + 1E-4;
    double z_upper =  1 - 1E-4;

    double a_lower = 0.8;
    double a_upper = 1.2;

    // Note: grid lengths must be even (edge case not handled)

    if(argc < 2)
    {
        std::cout << "Need to specify output path" << std::endl;
    }

    int numThreads = NUM_THREADS;
    int lenX = 14;
    int lenZ = 20;
    int lenA = 3;

    ScatteringProcessHandler<QuarkExchange> scatteringProcessHandler(numThreads, lenX, lenZ, lenA, 120, 40, 10, 10,
                                                                     eta, X_lower, X_upper,
                                                                     z_lower, z_upper, a_lower, a_upper, M_nucleon);
    scatteringProcessHandler.calculateScattering(1E3);
    scatteringProcessHandler.store_scattering_amplitude(argv[1]);

    return 0;
}
