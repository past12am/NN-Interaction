//
// Created by past12am on 8/18/23.
//

#include "../../include/scattering/ScatteringProcess.hpp"


ScatteringProcess::ScatteringProcess(int lenTau, int lenZ, double tauCutoff, double m, double M) : externalImpulseGrid(lenTau, lenZ, tauCutoff, m, M),
                                                                                                   tensorBasis(&externalImpulseGrid)
{
}
