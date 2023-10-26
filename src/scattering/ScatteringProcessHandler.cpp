//
// Created by past12am on 10/7/23.
//

#include <fstream>
#include "../../include/scattering/ScatteringProcessHandler.hpp"
#include "../../include/scattering/processes/QuarkExchange.hpp"

template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::ScatteringProcessHandler(int numThreads, int lenX, int lenZ, int lenA,
                                                                   int l2Points, int zPoints, int yPoints, int phiPoints,
                                                                   double eta,
                                                                   double XCutoffLower, double XCutoffUpper,
                                                                   double zCutoffLower, double zCutoffUpper,
                                                                   double aCutoffLower, double aCutoffUpper,
                                                                   const gsl_complex nucleonMass) :
        numThreads(numThreads), lenX(lenX), lenZ(lenZ), lenA(lenA), l2Points(l2Points), zPoints(zPoints), yPoints(yPoints),
        phiPoints(phiPoints), eta(eta), aCutoffLower(aCutoffLower), aCutoffUpper(aCutoffUpper), nucleon_mass(nucleonMass)
{
    subgridScatteringProcess = new ScatteringType*[numThreads * lenA];
    subgridIntegrationThread = new std::thread*[numThreads * lenA];


    a = new double[lenA];

    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        a[aIdx] = calcAAt(aIdx);
    }

    assert(a[0] == aCutoffLower);
    assert(a[lenA - 1] == aCutoffUpper);



    ZXGrid completeZXGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper);

    int numXPerThread = lenX / numThreads;
    if(lenX % numThreads != 0)
    {
        std::cout << "length of X grid must be divisible by number of threads" << std::endl;
        exit(-1);
    }

    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            int scatteringProcessIdx = calcScatteringProcessIdx(aIdx, threadIdx);

            double curXCutoffLower = completeZXGrid.getXAt(threadIdx * numXPerThread);
            double curXCutoffUpper = completeZXGrid.getXAt((threadIdx + 1) * numXPerThread - 1);

            if(typeid(ScatteringType) == typeid(QuarkExchange))
            {
                subgridScatteringProcess[scatteringProcessIdx] = new QuarkExchange(numXPerThread, lenZ, curXCutoffLower, curXCutoffUpper, zCutoffLower,
                                                                                   zCutoffUpper, nucleon_mass, a[aIdx], l2Points, zPoints, yPoints, phiPoints,
                                                                                   gsl_complex_rect(0.19, 0), eta, threadIdx);

            }
            else
            {
                std::cout << "Could not determine Scattering Type" << std::endl;
                exit(-1);
            }
        }
    }
}


template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::~ScatteringProcessHandler()
{
    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            delete subgridScatteringProcess[calcScatteringProcessIdx(aIdx, threadIdx)];
        }
    }

    delete[] a;
    delete[] subgridScatteringProcess;
}

template<class ScatteringType>
void ScatteringProcessHandler<ScatteringType>::calculateScattering(double l2_cutoff)
{
    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        std::cout << "Calculating for aIdx [" << aIdx << "/" << lenA << "]" << std::endl;
        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            int scatteringIdx = calcScatteringProcessIdx(aIdx, threadIdx);
            subgridIntegrationThread[scatteringIdx] = new std::thread(&QuarkExchange::performScatteringCalculation,
                                                                  ((QuarkExchange*) subgridScatteringProcess[scatteringIdx]),
                                                                  l2_cutoff);
        }

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            subgridIntegrationThread[calcScatteringProcessIdx(aIdx, threadIdx)]->join();
        }
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

        data_file << "a,X,z,PK,QQ,h,f,|scattering_amp|2" << std::endl;

        for(int aIdx = 0; aIdx < lenA; aIdx++)
        {
            for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
            {
                ((ScatteringProcess*) subgridScatteringProcess[calcScatteringProcessIdx(aIdx, threadIdx)])
                        ->store_scattering_amplitude(basisElemIdx, a[aIdx], data_file);
            }
        }
    }
}


template<class ScatteringType>
double ScatteringProcessHandler<ScatteringType>::calcAAt(int aIdx)
{
    return aCutoffLower + (aCutoffUpper - aCutoffLower) * ((double) aIdx) / ((double) (lenA - 1));
}

template<class ScatteringType>
int ScatteringProcessHandler<ScatteringType>::calcScatteringProcessIdx(int aIdx, int threadIdx)
{
    return threadIdx + aIdx * numThreads;
}

template class ScatteringProcessHandler<QuarkExchange>;