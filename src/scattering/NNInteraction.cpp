//
// Created by past12am on 8/18/23.
//

#include "../../include/scattering/NNInteraction.hpp"


NNInteraction::NNInteraction(ScatteringProcess* scattering, gsl_complex M_nucleon, int lenZ, int lenTau,
                             double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper) : scattering(scattering), externalImpulseGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper, M_nucleon), tensorBasis(&externalImpulseGrid)
{
    // define impulses




}

void NNInteraction::calcScatteringAmplitude()
{

}
