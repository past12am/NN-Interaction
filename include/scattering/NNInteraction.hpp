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
        NNInteraction(ScatteringProcess* scattering, gsl_complex M_nucleon, int lenZ, int lenTau, double tauCutoffLower, double tauCutoffUpper);

        void calcScatteringAmplitude();
};


#endif //NNINTERACTION_NNINTERACTION_HPP
