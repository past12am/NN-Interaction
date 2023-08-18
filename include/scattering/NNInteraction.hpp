//
// Created by past12am on 8/18/23.
//

#ifndef NNINTERACTION_NNINTERACTION_HPP
#define NNINTERACTION_NNINTERACTION_HPP


#include "ScatteringProcess.hpp"
#include "basis/TensorBasis.hpp"

class NNInteraction
{
    private:
        ScatteringProcess* scattering;
        TensorBasis tensorBasis;
        ExternalImpulseGrid externalImpulseGrid;

    public:
        NNInteraction(ScatteringProcess* scattering, double m, double M, int lenZ, int lenTau, double tauCutoff);

        void calcScatteringAmplitude();
};


#endif //NNINTERACTION_NNINTERACTION_HPP
