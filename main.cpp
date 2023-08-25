#include <iostream>
#include "include/scattering/basis/TensorBasis.hpp"
#include "include/scattering/impulse/ExternalImpulseGrid.hpp"
#include "include/scattering/ScatteringProcess.hpp"
#include "include/scattering/processes/QuarkExchange.hpp"

#include <complex>
#include <gsl/gsl_math.h>


int main()
{
    // define impulses
    double m = 0.004; // GeV
    double M = 0.007; // GeV

    double p2_ir_cutoff = 1E-2;
    double p2_uv_cutoff = 3E3;
    double p2_mid_cutoff = 600;


    // Note: grid lengths must be even (edge case not handled)
    ScatteringProcess* scattering = new QuarkExchange(100, 30, 3, m, M,
                                        100, 30, 30, 10, gsl_complex_rect(0.19, 0));


    return 0;
}
