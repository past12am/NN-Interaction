#include <iostream>
#include "include/Definitions.h"
#include "include/scattering/basis/TensorBasis.hpp"
#include "include/scattering/impulse/ExternalImpulseGrid.hpp"
#include "include/scattering/ScatteringProcess.hpp"
#include "include/scattering/processes/QuarkExchange.hpp"
#include "include/scattering/ScatteringProcessHandler.hpp"
#include "include/data/QuarkDiquarkAmplitudeReader.hpp"

#include <complex>
#include <gsl/gsl_math.h>


/*
NN Interaction - To do:
    both things on the same footing
        -> Add Diquark, Pion, Scalar Exchange
        -> do contour deformations


    phase shifts --> not really possible (without extension)

    Format: Phys Let B, or normal papers (Phys Rev. D)


    Note: if problem with contour def. just use mass pole


    Answer the sign question
 */

/*
 *  TODO: improve multithreading (make independent of lenX --> use "parallel integrators" in a threadpool fashion instead)
 *
 */


int main(int argc, char *argv[])
{
    double m_q = 0.55; // GeV   // TODO 0.5 GeV?
    double m_d = 0.8;  // GeV (scalar diquark)

    double eta = m_q/(m_q + m_d);

    //double impulse_ir_cutoff = 1E-2;
    //double impulse_uv_cutoff = 3E3;
    //double impulse_mid_cutoff = 600;

    double X_upper = 0.99;
    double X_lower = 0.1;

    double Z_lower = -1; // -1 + 1E-4;
    double Z_upper = 1; // 1 - 1E-4;

    double contour_def_eps_lower = 0.01;
    double contour_def_eps_upper = 0.1;


    // Note: grid lengths for X must be even (edge case not handled)
    //       grid lengths for Z must be odd, s.t. 0 is included

    if(argc < 2)
    {
        std::cout << "Need to specify output path" << std::endl;
    }

    int numThreads = NUM_THREADS;
    int lenX = 8;
    int lenZ = 11;
    int lenContourDef = 6;


    int k2_integration_points = 60;
    int z_integration_points = 60;
    int y_integration_points = 32;
    int phi_integration_points = 20;


    /* Working parameter sets
     * Mini (only kinematically safe contour)
    *    int k2_integration_points = 20;
    *    int z_integration_points = 20;
    *    int y_integration_points = 20;
    *    int phi_integration_points = 20;
    *
    *    #define CUTOFF_k2 1E4
    *
     * Small
    *    int k2_integration_points = 60;
    *    int z_integration_points = 60;
    *    int y_integration_points = 32;
    *    int phi_integration_points = 20;
    *
    *    #define CUTOFF_x_4 30.0
    *    #define CUTOFF_absx 30.0
    *
     * Medium
    *    int k2_integration_points = 80;
    *    int z_integration_points = 80;
    *    int y_integration_points = 32;
    *    int phi_integration_points = 20;
    *
    *    #define CUTOFF_x_4 40.0
    *    #define CUTOFF_absx 40.0
    */
    // Sanity Checks for Parameters
    //      we need ANALYTIC --> BASIS = tau, PROJECTION_BASIS = tau_prime
    if(INVERT_STRATEGY == InvertStrategy::ANALYTIC)
    {
        if(!(PROJECTION_BASIS == Basis::tau_prime && BASIS == Basis::tau))
        {
            std::cout << "Invalid combination of invert strategy and basis for dressing function, we need --> BASIS = tau, PROJECTION_BASIS = tau_prime" << std::endl;
            exit(2);
        }
    }
    else if (INVERT_STRATEGY == InvertStrategy::NUMERIC_MATRIX_INVERSE)
    {
        if(PROJECTION_BASIS != BASIS)
        {
            std::cout << "Need Projection Basis == Basis" << std::endl;
            exit(2);
        }
    }


    // Set Data path for fits
    QuarkDiquarkAmplitudeReader::setPath(argv[2]);


    // Do calculation
    ScatteringProcessHandler scatteringProcessHandler(numThreads, lenX, lenZ, lenContourDef,
                                                      k2_integration_points, z_integration_points,
                                                      y_integration_points, phi_integration_points,
                                                      eta, X_lower, X_upper,
                                                      Z_lower, Z_upper,
                                                      contour_def_eps_lower, contour_def_eps_upper);

    scatteringProcessHandler.calculateScattering();
    scatteringProcessHandler.store_scattering_amplitude(argv[1],
                                                        lenX,
                                                        lenZ,
                                                        lenContourDef,
                                                        X_lower,
                                                        X_upper,
                                                        Z_lower,
                                                        Z_upper,
                                                        k2_integration_points,
                                                        z_integration_points,
                                                        y_integration_points,
                                                        phi_integration_points);

    return 0;
}
