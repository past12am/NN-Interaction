#include <iostream>
#include "include/scattering/basis/TensorBasis.hpp"
#include "include/helpers/ExternalImpulseGrid.hpp"
#include "include/scattering/ScatteringProcess.hpp"
#include "include/scattering/processes/QuarkExchange.hpp"

#include <complex>
#include <gsl/gsl_math.h>


int main()
{
    // define impulses
    double m = 0.004; // GeV
    double M = 0.007; // GeV


    ScatteringProcess* scattering = new QuarkExchange(100, 30, 3, m, M);


    return 0;
}
