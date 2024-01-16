//
// Created by past12am on 10/7/23.
//

#ifndef NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
#define NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP


#include <thread>
#include "ScatteringProcess.hpp"


class ScatteringProcessHandler
{
    private:
        int numThreads;

        int lenX;
        int lenZ;

        int k2Points;
        int zPoints;
        int yPoints;
        int phiPoints;

        double eta;

        double aCutoffLower;
        double aCutoffUpper;

        gsl_complex nucleon_mass;

        std::thread** subgridIntegrationThread;
        ScatteringProcess** subgridScatteringProcess;


    public:
        ScatteringProcessHandler(int numThreads, int lenX, int lenZ, int k2Points, int zPoints, int yPoints,
                                 int phiPoints, double eta, double XCutoffLower, double XCutoffUpper,
                                 double ZCutoffLower, double ZCutoffUpper, gsl_complex nucleonMass);

        virtual ~ScatteringProcessHandler();

        void calculateScattering(double k2_cutoff);
        void store_scattering_amplitude(std::string data_path,
                                        int lenX,
                                        int lenZ,
                                        double X_lower,
                                        double X_upper,
                                        double Z_lower,
                                        double Z_upper,
                                        double loop_cutoff,
                                        int k2_integration_points,
                                        int z_integration_points,
                                        int y_integration_points,
                                        int phi_integration_points);
};


#endif //NNINTERACTION_SCATTERINGPROCESSHANDLER_HPP
