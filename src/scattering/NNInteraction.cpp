//
// Created by past12am on 8/18/23.
//

#include "../../include/scattering/NNInteraction.hpp"


NNInteraction::NNInteraction(ScatteringProcess* scattering, double m, double M, int lenZ, int lenTau, double tauCutoff) : scattering(scattering), externalImpulseGrid(lenTau, lenZ, tauCutoff, m, M),
                                                                                                                          tensorBasis(&externalImpulseGrid)
{
    // define impulses




}

void NNInteraction::calcScatteringAmplitude()
{

}
