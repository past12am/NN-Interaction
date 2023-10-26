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

        int lenX;
        int lenZ;
        int lenA;

        int l2Points;
        int zPoints;
        int yPoints;
        int phiPoints;

        double eta;

        double aCutoffLower;
        double aCutoffUpper;

        gsl_complex nucleon_mass;
        double* a;

        std::thread** subgridIntegrationThread;
        ScatteringType** subgridScatteringProcess;

        double calcAAt(int aIdx);

        int calcScatteringProcessIdx(int aIdx, int threadIdx);


    public:
        ScatteringProcessHandler(int numThreads, int lenX, int lenZ, int lenA, int l2Points, int zPoints, int yPoints,
                                 int phiPoints, double eta, double XCutoffLower, double XCutoffUpper,
                                 double zCutoffLower, double zCutoffUpper, double aCutoffLower, double aCutoffUpper, gsl_complex nucleonMass);

        virtual ~ScatteringProcessHandler();

        void calculateScattering(double l2_cutoff);
        void store_scattering_amplitude(std::string data_path);
};


#endif //NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
