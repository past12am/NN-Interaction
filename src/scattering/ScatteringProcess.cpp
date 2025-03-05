//
// Created by past12am on 8/18/23.
//

#include <fstream>
#include <gsl/gsl_blas.h>
#include <cassert>
#include <iostream>
#include <chrono>

#include "../../include/scattering/ScatteringProcess.hpp"
#include "../../include/Definitions.h"
#include "../../include/utils/print/PrintGSLElements.hpp"


ScatteringProcess::ScatteringProcess(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, int threadIdx) :
        threadIdx(threadIdx), nucleon_mass(nucleon_mass),
        externalImpulseGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper, nucleon_mass),
        tensorBasis(&externalImpulseGrid, nucleon_mass)
{
    scattering_amplitude_basis_projected = new gsl_complex[tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength()];
    form_factors = new gsl_complex[tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength()];

    inverseKMatrix = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_set_zero(inverseKMatrix);

    scattering_amplitude = new Tensor4<4, 4, 4, 4>[externalImpulseGrid.getLength()];
    for(int i = 0; i < externalImpulseGrid.getLength(); i++)
    {
        scattering_amplitude[i] = Tensor4<4, 4, 4, 4>();
    }

    k = gsl_vector_complex_alloc(4);
}

ScatteringProcess::~ScatteringProcess()
{
    gsl_vector_complex_free(k);

    delete[] scattering_amplitude;

    gsl_matrix_complex_free(inverseKMatrix);

    delete[] form_factors;
    delete[] scattering_amplitude_basis_projected;
}

gsl_complex ScatteringProcess::integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double k2, double z, double y, double phi)
{
    if(!k_mutex.try_lock())
    {
        std::cout << "Probable race condition on temporary k impulse" << std::endl;
        exit(-1);
    }

    // Set current k
    gsl_vector_complex_set_zero(k);
    momentumLoop->calc_k(k2, z, y, phi, k);

    // get basis Element
    Tensor4<4, 4, 4, 4>* currentBasisProjectionElement = tensorBasis.basisTensorProjection(basisElemIdx, externalImpulseIdx);

    // get Tensor
    Tensor4<4, 4, 4, 4> integralKernelTensor = Tensor4<4, 4, 4, 4>();
    integralKernel(k,
                   externalImpulseGrid.get_l_ext(externalImpulseIdx), externalImpulseGrid.get_r_ext(externalImpulseIdx),
                   externalImpulseGrid.get_P_ext(externalImpulseIdx),
                   externalImpulseGrid.get_p_f(externalImpulseIdx), externalImpulseGrid.get_p_i(externalImpulseIdx),
                   externalImpulseGrid.get_k_f(externalImpulseIdx), externalImpulseGrid.get_k_i(externalImpulseIdx),
                   &integralKernelTensor);

    k_mutex.unlock();

    gsl_complex kernel_res = integralKernelTensor.leftContractWith(currentBasisProjectionElement);
    return kernel_res;
}

void ScatteringProcess::integrate(double k2_cutoff)
{
    int num_progress_char = 100;
    int progress = 0;
    int total = tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength();


    std::chrono::time_point clock_at_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point clock_at_end = std::chrono::high_resolution_clock::now();
    auto avg_time = clock_at_end - clock_at_start;


    std::cout << "Thread " << threadIdx << " calculates " << tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength() << " grid points" << std::endl;

    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
    {
        // Integrate each Scattering Matrix element for each choice of external Impulse
        for (int externalImpulseIdx = 0; externalImpulseIdx < externalImpulseGrid.getLength(); externalImpulseIdx++)
        {
            progress = basisElemIdx * externalImpulseGrid.getLength() + externalImpulseIdx;
            avg_time = (clock_at_end - clock_at_start)/(progress + 1);

            std::cout << "Thread " << threadIdx << "-->   Basis Element [" << std::string(((double) progress/total) * num_progress_char, '#') << std::string((1.0 - (double) progress/total) * num_progress_char, '-').c_str() << "]    --> "
                      << std::chrono::duration_cast<std::chrono::minutes>(clock_at_end - clock_at_start) << " of " << std::chrono::duration_cast<std::chrono::minutes>(avg_time * total) << "\t" << std::flush;


            gsl_complex res = integrate_process(basisElemIdx, externalImpulseIdx, k2_cutoff);
            scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)] = res;

            std::cout << "Basis[" << basisElemIdx << "], basisIdx=" << externalImpulseIdx << ": " << GSL_REAL(res) << " + i " << GSL_IMAG(res) << std::endl;

            clock_at_end = std::chrono::high_resolution_clock::now();
        }
    }
}

void ScatteringProcess::store_scattering_amplitude(int basisElemIdx, std::ofstream& data_file)
{
    for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
    {
        double X = externalImpulseGrid.calcXAt(XIdx);

        for(int ZIdx = 0; ZIdx < externalImpulseGrid.getLenZ(); ZIdx++)
        {
            double Z = externalImpulseGrid.calcZAt(ZIdx);

            int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

            gsl_complex h_i = scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)];
            gsl_complex f_i = form_factors[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)];

            data_file << X << "," << Z << ","
                      << GSL_REAL(h_i) << (GSL_IMAG(h_i) < 0 ? "-" : "+") << abs(GSL_IMAG(h_i)) << "i" << ","
                      << GSL_REAL(f_i) << (GSL_IMAG(f_i) < 0 ? "-" : "+") << abs(GSL_IMAG(f_i)) << "i" << ","
                      << calcSquaredNormOfScatteringMatrix(externalImpulseIdx) << std::endl;
        }
    }
}

int ScatteringProcess::calcScatteringAmpIdx(int basisElemIdx, int externalImpulseIdx)
{
    return basisElemIdx * externalImpulseGrid.getLength() + externalImpulseIdx;
}

double ScatteringProcess::calcSquaredNormOfScatteringMatrix(int externalImpulseIdx)
{
    double squared_scattering_matrix_elem = scattering_amplitude[externalImpulseIdx].absSquare();
    return squared_scattering_matrix_elem;
}

void ScatteringProcess::build_h_vector(int externalImpulseIdx, gsl_vector_complex* h)
{
    gsl_vector_complex_set_zero(h);

    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
    {
        int scatteringAmpIdx = calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx);
        gsl_vector_complex_set(h, basisElemIdx, scattering_amplitude_basis_projected[scatteringAmpIdx]);
    }
}

void ScatteringProcess::calculateFormFactors(int XIdx, int ZIdx, gsl_complex M, gsl_vector_complex* f)
{
    int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

    gsl_vector_complex* h = gsl_vector_complex_alloc(tensorBasis.getTensorBasisElementCount());
    build_h_vector(externalImpulseIdx, h);

    gsl_matrix_complex* h2fMatrix;

    if(INVERT_STRATEGY == InvertStrategy::NUMERIC_MATRIX_INVERSE)
    {
        gsl_matrix_complex* invK = tensorBasis.KInv(externalImpulseIdx);
        h2fMatrix = invK;
    }
    else if(INVERT_STRATEGY == InvertStrategy::ANALYTIC)
    {
        gsl_matrix_complex* invR = tensorBasis.RInv(externalImpulseIdx);
        h2fMatrix = invR;
    }
    else
    {
        // If it happens, we fucked up the configuration (no cleaner exit needed)
        exit(2);
    }


    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, h2fMatrix, h, GSL_COMPLEX_ZERO, f);
    gsl_vector_complex_free(h);
}

void ScatteringProcess::buildScatteringMatrix()
{
    gsl_vector_complex* f = gsl_vector_complex_alloc(tensorBasis.getTensorBasisElementCount());

    for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
    {
        for (int ZIdx = 0; ZIdx < externalImpulseGrid.getLenZ(); ZIdx++)
        {
            int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

            // find f
            gsl_vector_complex_set_zero(f);
            calculateFormFactors(XIdx, ZIdx, nucleon_mass, f);

            // loop over tensor basis
            scattering_amplitude[externalImpulseIdx].setZero();
            for (int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
            {
                form_factors[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)] = gsl_vector_complex_get(f, basisElemIdx);

                scattering_amplitude[externalImpulseIdx] += (*tensorBasis.basisTensor(basisElemIdx, externalImpulseIdx)) * gsl_vector_complex_get(f, basisElemIdx);
            }
        }
    }
}

void ScatteringProcess::performScatteringCalculation(double k2_cutoff)
{
    integrate(k2_cutoff);
    buildScatteringMatrix();
}

TensorBasis* ScatteringProcess::getTensorBasis()
{
    return &tensorBasis;
}
