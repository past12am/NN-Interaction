//
// Created by past12am on 10/7/23.
//

#include <fstream>
#include "../../include/scattering/ScatteringProcessHandler.hpp"
#include "../../include/scattering/processes/QuarkExchange.hpp"
#include "../../include/Definitions.h"

template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::ScatteringProcessHandler(int numThreads, int lenX, int lenZ,
                                                                   int l2Points, int zPoints, int yPoints, int phiPoints,
                                                                   double eta,
                                                                   double XCutoffLower, double XCutoffUpper,
                                                                   double ZCutoffLower, double ZCutoffUpper,
                                                                   const gsl_complex nucleonMass) :
        numThreads(numThreads), lenX(lenX), lenZ(lenZ), l2Points(l2Points), zPoints(zPoints), yPoints(yPoints),
        phiPoints(phiPoints), eta(eta), nucleon_mass(nucleonMass)
{
    subgridScatteringProcess = new ScatteringType*[numThreads];
    subgridIntegrationThread = new std::thread*[numThreads];



    ZXGrid completeZXGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, ZCutoffLower, ZCutoffUpper);

    int numXPerThread = lenX / numThreads;
    if(lenX % numThreads != 0)
    {
        std::cout << "length of X grid must be divisible by number of threads" << std::endl;
        exit(-1);
    }


    for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        double curXCutoffLower = completeZXGrid.getXAt(threadIdx * numXPerThread);
        double curXCutoffUpper = completeZXGrid.getXAt((threadIdx + 1) * numXPerThread - 1);

        if(typeid(ScatteringType) == typeid(QuarkExchange))
        {
            subgridScatteringProcess[threadIdx] = new QuarkExchange(numXPerThread, lenZ,
                                                                    curXCutoffLower, curXCutoffUpper,
                                                                    ZCutoffLower, ZCutoffUpper,
                                                                    nucleon_mass, eta,
                                                                    l2Points, zPoints, yPoints, phiPoints,
                                                                    threadIdx);
        }
        else
        {
            std::cout << "Could not determine Scattering Type" << std::endl;
            exit(-1);
        }
    }
}


template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::~ScatteringProcessHandler()
{
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        delete subgridScatteringProcess[threadIdx];
    }

    delete[] subgridScatteringProcess;
}

template<class ScatteringType>
void ScatteringProcessHandler<ScatteringType>::calculateScattering(double k2_cutoff)
{
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        subgridIntegrationThread[threadIdx] = new std::thread(&QuarkExchange::performScatteringCalculation,
                                                                  ((QuarkExchange*) subgridScatteringProcess[threadIdx]),
                                                                  k2_cutoff);
    }

    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        subgridIntegrationThread[threadIdx]->join();
    }
}

template<class ScatteringType>
void ScatteringProcessHandler<ScatteringType>::store_scattering_amplitude(std::string data_path)
{
    for(int basisElemIdx = 0; basisElemIdx < ((ScatteringProcess*) subgridScatteringProcess[0])->getTensorBasis()->getTensorBasisElementCount(); basisElemIdx++)
    {
        std::ostringstream fnamestrstream;
        fnamestrstream << data_path;

        if(BASIS == 0)
            fnamestrstream << "/tau_";
        else if(BASIS == 1)
            fnamestrstream << "/T_";
        else
        {
            std::cout << "Unknown Basis " << BASIS << std::endl;
            exit(-1);
        }

        fnamestrstream << basisElemIdx << ".txt";


        std::ofstream data_file;
        data_file.open(fnamestrstream.str(), std::ofstream::out | std::ios::trunc);

        data_file << "X,Z,h,f,|scattering_amp|2" << std::endl;

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            ((ScatteringProcess*) subgridScatteringProcess[threadIdx])
                    ->store_scattering_amplitude(basisElemIdx, data_file);
        }
    }
}

template class ScatteringProcessHandler<QuarkExchange>;