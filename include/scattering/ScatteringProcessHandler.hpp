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

        int l2Points;
        int zPoints;
        int yPoints;
        int phiPoints;

        double eta;

        double tauCutoffLower;
        double tauCutoffUpper;

        double zCutoffLower;
        double zCutoffUpper;

        gsl_complex nucleon_mass;

        std::thread** subgridIntegrationThread;
        ScatteringType** subgridScatteringProcess;


    public:
        ScatteringProcessHandler(int numThreads, int lenTau, int lenZ, int l2Points, int zPoints, int yPoints,
                                 int phiPoints, double eta, double tauCutoffLower, double tauCutoffUpper,
                                 double zCutoffLower, double zCutoffUpper, gsl_complex nucleonMass);

        virtual ~ScatteringProcessHandler();

        void calculateScattering(double l2_cutoff);
        void store_scattering_amplitude(std::string data_path);
};


#endif //NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
