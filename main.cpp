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
    int lenZ = 12;

    double loop_cutoff = 1E4;

    int k2_integration_points = 32;
    int z_integration_points = 16;
    int y_integration_points = 16;
    int phi_integration_points = 10;

    ScatteringProcessHandler scatteringProcessHandler(numThreads, lenX, lenZ,
                                                      k2_integration_points, z_integration_points,
                                                      y_integration_points, phi_integration_points,
                                                      eta, X_lower, X_upper,
                                                      Z_lower, Z_upper, M_nucleon);

    scatteringProcessHandler.calculateScattering(loop_cutoff);
    scatteringProcessHandler.store_scattering_amplitude(argv[1],
                                                        lenX,
                                                        lenZ,
                                                        X_lower,
                                                        X_upper,
                                                        Z_lower,
                                                        Z_upper,
                                                        loop_cutoff,
                                                        k2_integration_points,
                                                        z_integration_points,
                                                        y_integration_points,
                                                        phi_integration_points);

    return 0;
}
