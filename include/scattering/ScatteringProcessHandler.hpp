//
// Created by past12am on 10/7/23.
//

#ifndef NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
#define NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP


#include <thread>
#include "ScatteringProcess.hpp"

template<class ScatteringType>
class ScatteringProcessHandler
{
    private:
        int numThreads;

        int lenTau;
        int lenZ;
        int lenA;

        int l2Points;
        int zPoints;
        int yPoints;
        int phiPoints;

        double eta;

        double aImagCutoffLower;
        double aImagCutoffUpper;

        gsl_complex nucleon_mass;
        gsl_complex* a;

        std::thread** subgridIntegrationThread;
        ScatteringType** subgridScatteringProcess;

        gsl_complex calcAAt(int aIdx);

        int calcScatteringProcessIdx(int aIdx, int threadIdx);


    public:
        ScatteringProcessHandler(int numThreads, int lenTau, int lenZ, int lenA, int l2Points, int zPoints, int yPoints,
                                 int phiPoints, double eta, double tauCutoffLower, double tauCutoffUpper,
                                 double zCutoffLower, double zCutoffUpper, double aImagCutoffLower, double aImagCutoffUpper, gsl_complex nucleonMass);

        virtual ~ScatteringProcessHandler();

        void calculateScattering(double l2_cutoff);
        void store_scattering_amplitude(std::string data_path);
};


#endif //NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
