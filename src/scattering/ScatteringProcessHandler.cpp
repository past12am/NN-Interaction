//
// Created by past12am on 10/7/23.
//

#include <fstream>
#include "../../include/scattering/ScatteringProcessHandler.hpp"
#include "../../include/scattering/processes/QuarkExchange.hpp"

template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::ScatteringProcessHandler(int numThreads, int lenTau, int lenZ, int l2Points,
                                                                   int zPoints, int yPoints, int phiPoints, double eta,
                                                                   double tauCutoffLower, double tauCutoffUpper,
                                                                   double zCutoffLower, double zCutoffUpper,
                                                                   const gsl_complex nucleonMass) :
        numThreads(numThreads), lenTau(lenTau), lenZ(lenZ), l2Points(l2Points), zPoints(zPoints), yPoints(yPoints),
        phiPoints(phiPoints), eta(eta), tauCutoffLower(tauCutoffLower), tauCutoffUpper(tauCutoffUpper),
        zCutoffLower(zCutoffLower), zCutoffUpper(zCutoffUpper), nucleon_mass(nucleonMass)
{
    subgridScatteringProcess = new ScatteringType*[numThreads];
    subgridIntegrationThread = new std::thread*[numThreads];

    ZTauGrid completeZTauGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper);

    int numTauPerThread = lenTau / numThreads;
    if(lenTau % numThreads != 0)
    {
        std::cout << "length of tau grid must be divisible by number of threads" << std::endl;
        exit(-1);
    }

    for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        double curTauCutoffLower = completeZTauGrid.getTauAt(threadIdx * numTauPerThread);
        double curTauCutoffUpper = completeZTauGrid.getTauAt((threadIdx + 1) * numTauPerThread - 1);


        if(typeid(ScatteringType) == typeid(QuarkExchange))
        {
            subgridScatteringProcess[threadIdx] = new QuarkExchange(numTauPerThread, lenZ, curTauCutoffLower, curTauCutoffUpper, zCutoffLower,
                                                                    zCutoffUpper, nucleon_mass, l2Points, zPoints, yPoints, phiPoints,
                                                                    gsl_complex_rect(0.19, 0), eta, threadIdx);

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
    for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        delete subgridScatteringProcess[threadIdx];
    }

    delete[] subgridScatteringProcess;
}

template<class ScatteringType>
void ScatteringProcessHandler<ScatteringType>::calculateScattering(double l2_cutoff)
{
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        subgridIntegrationThread[threadIdx] = new std::thread(&QuarkExchange::performScatteringCalculation, ((QuarkExchange*) subgridScatteringProcess[threadIdx]), l2_cutoff);
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
        fnamestrstream << data_path << "/tau_" << basisElemIdx << ".txt";

        std::ofstream data_file;
        data_file.open(fnamestrstream.str(), std::ofstream::out | std::ios::trunc);

        data_file << "tau,z,PK,QQ,h,f,|scattering_amp|2" << std::endl;

        for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            ((ScatteringProcess*) subgridScatteringProcess[threadIdx])->store_scattering_amplitude(basisElemIdx, data_file);
        }
    }
}

template class ScatteringProcessHandler<QuarkExchange>;