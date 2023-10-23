//
// Created by past12am on 10/7/23.
//

#include <fstream>
#include "../../include/scattering/ScatteringProcessHandler.hpp"
#include "../../include/scattering/processes/QuarkExchange.hpp"

template<class ScatteringType>
ScatteringProcessHandler<ScatteringType>::ScatteringProcessHandler(int numThreads, int lenTau, int lenZ, int lenA,
                                                                   int l2Points, int zPoints, int yPoints, int phiPoints,
                                                                   double eta,
                                                                   double tauCutoffLower, double tauCutoffUpper,
                                                                   double zCutoffLower, double zCutoffUpper,
                                                                   double aImagCutoffLower, double aImagCutoffUpper,
                                                                   const gsl_complex nucleonMass) :
        numThreads(numThreads), lenTau(lenTau), lenZ(lenZ), lenA(lenA), l2Points(l2Points), zPoints(zPoints), yPoints(yPoints),
        phiPoints(phiPoints), eta(eta), aImagCutoffLower(aImagCutoffLower), aImagCutoffUpper(aImagCutoffUpper), nucleon_mass(nucleonMass)
{
    subgridScatteringProcess = new ScatteringType*[numThreads * lenA];
    subgridIntegrationThread = new std::thread*[numThreads * lenA];


    a = new gsl_complex[lenA];

    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        a[aIdx] = calcAAt(aIdx);
    }

    assert(GSL_IMAG(a[0]) == aImagCutoffLower);
    assert(GSL_IMAG(a[lenA - 1]) == aImagCutoffUpper);



    ZTauGrid completeZTauGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper);

    int numTauPerThread = lenTau / numThreads;
    if(lenTau % numThreads != 0)
    {
        std::cout << "length of tau grid must be divisible by number of threads" << std::endl;
        exit(-1);
    }

    for(int aIdx = 0; aIdx < lenA; aIdx++)
    {
        for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            int scatteringProcessIdx = calcScatteringProcessIdx(aIdx, threadIdx);

            double curTauCutoffLower = completeZTauGrid.getTauAt(threadIdx * numTauPerThread);
            double curTauCutoffUpper = completeZTauGrid.getTauAt((threadIdx + 1) * numTauPerThread - 1);

            if(typeid(ScatteringType) == typeid(QuarkExchange))
            {
                subgridScatteringProcess[scatteringProcessIdx] = new QuarkExchange(numTauPerThread, lenZ, curTauCutoffLower, curTauCutoffUpper, zCutoffLower,
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

        data_file << "a,tau,z,PK,QQ,h,f,|scattering_amp|2" << std::endl;

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
gsl_complex ScatteringProcessHandler<ScatteringType>::calcAAt(int aIdx)
{
    double aImag = aImagCutoffLower + (aImagCutoffUpper - aImagCutoffLower) * ((double) aIdx)/((double) (lenA - 1));
    return gsl_complex_rect(0, aImag);
}

template<class ScatteringType>
int ScatteringProcessHandler<ScatteringType>::calcScatteringProcessIdx(int aIdx, int threadIdx)
{
    return threadIdx + aIdx * numThreads;
}

template class ScatteringProcessHandler<QuarkExchange>;