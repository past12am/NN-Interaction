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
    double m_q = 0.55; // GeV
    double m_d = 0.8;  // GeV (scalar diquark)
    gsl_complex M_nucleon = gsl_complex_rect(0.94, 0); // GeV

    double eta = m_q/(m_q + m_d);

    //double impulse_ir_cutoff = 1E-2;
    //double impulse_uv_cutoff = 3E3;
    //double impulse_mid_cutoff = 600;

    double X_upper = 0.99;
    double X_lower = 0.1;

    double Z_lower = -1; // -1 + 1E-4;
    double Z_upper = 1; // 1 - 1E-4;


    // Note: grid lengths must be even (edge case not handled)

    if(argc < 2)
    {
        std::cout << "Need to specify output path" << std::endl;
    }

    int numThreads = NUM_THREADS;
    int lenX = 12;
    int lenZ = 15;

    ScatteringProcessHandler<QuarkExchange> scatteringProcessHandler(numThreads, lenX, lenZ, 200, 40, 7, 7,
                                                                     eta, X_lower, X_upper,
                                                                     Z_lower, Z_upper, M_nucleon);
    scatteringProcessHandler.calculateScattering(1E4);
    scatteringProcessHandler.store_scattering_amplitude(argv[1]);

    return 0;
}
