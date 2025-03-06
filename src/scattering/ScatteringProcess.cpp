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


ScatteringProcess::ScatteringProcess(int lenX, int lenZ, int lenContourDef, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, double contourEpsLower, double contourEpsUpper, int threadIdx) :
        threadIdx(threadIdx),
        len_contour_def_epsilon(lenContourDef),
        externalImpulseGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper),
        tensorBasis(&externalImpulseGrid)
{
    contour_def_epsilon = new double[len_contour_def_epsilon];
    for (int i = 0; i < len_contour_def_epsilon; ++i)
    {
        contour_def_epsilon[i] = len_contour_def_epsilon == 1 ? contourEpsLower : contourEpsLower + (contourEpsUpper - contourEpsLower) * ((double) i) / ((double) (len_contour_def_epsilon - 1));
    }

    scattering_amplitude_basis_projected = new gsl_complex[tensorBasis.getTensorBasisElementCount() * len_contour_def_epsilon * externalImpulseGrid.getLength()];
    form_factors = new gsl_complex[tensorBasis.getTensorBasisElementCount() * len_contour_def_epsilon * externalImpulseGrid.getLength()];

    inverseKMatrix = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_set_zero(inverseKMatrix);

    scattering_amplitude = new Tensor4<4, 4, 4, 4>[len_contour_def_epsilon * externalImpulseGrid.getLength()];
    for(int i = 0; i < len_contour_def_epsilon * externalImpulseGrid.getLength(); i++)
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

    delete[] contour_def_epsilon;
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
    int total = tensorBasis.getTensorBasisElementCount() * len_contour_def_epsilon * externalImpulseGrid.getLength();


    std::chrono::time_point clock_at_start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point clock_at_end = std::chrono::high_resolution_clock::now();
    auto avg_time = clock_at_end - clock_at_start;


    std::cout << "Thread " << threadIdx << " calculates " << tensorBasis.getTensorBasisElementCount() * len_contour_def_epsilon * externalImpulseGrid.getLength() << " grid points" << std::endl;

    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
    {
        for(int contour_def_idx = 0; contour_def_idx < len_contour_def_epsilon; contour_def_idx++)
        {
            // Integrate each Scattering Matrix element for each choice of external Impulse
            for (int externalImpulseIdx = 0; externalImpulseIdx < externalImpulseGrid.getLength(); externalImpulseIdx++)
            {
                progress = calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx); // basisElemIdx * len_contour_def_epsilon * externalImpulseGrid.getLength() + contour_def_idx * externalImpulseGrid.getLength() + externalImpulseIdx;
                avg_time = (clock_at_end - clock_at_start)/(progress + 1);

                std::cout << "Thread " << threadIdx << "-->   Basis Element [" << std::string(((double) progress/total) * num_progress_char, '#') << std::string((1.0 - (double) progress/total) * num_progress_char, '-').c_str() << "]    --> "
                          << std::chrono::duration_cast<std::chrono::minutes>(clock_at_end - clock_at_start) << " of " << std::chrono::duration_cast<std::chrono::minutes>(avg_time * total) << "\t" << std::flush;


                gsl_complex res = integrate_process(basisElemIdx, contour_def_idx, externalImpulseIdx, k2_cutoff);
                scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx)] = res;

                std::cout << "Basis[" << basisElemIdx << "], eps-idx=" << externalImpulseIdx << ", impulse-idx=" << externalImpulseIdx << ": " << GSL_REAL(res) << " + i " << GSL_IMAG(res) << std::endl;

                clock_at_end = std::chrono::high_resolution_clock::now();
            }
        }
    }
}

void ScatteringProcess::store_scattering_amplitude(int basisElemIdx, std::ofstream& data_file)
{
    for(int contour_def_idx = 0; contour_def_idx < len_contour_def_epsilon; contour_def_idx++)
    {
        double eps = contour_def_epsilon[contour_def_idx];

        for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
        {
            double X = externalImpulseGrid.calcXAt(XIdx);

            for(int ZIdx = 0; ZIdx < externalImpulseGrid.getLenZ(); ZIdx++)
            {
                double Z = externalImpulseGrid.calcZAt(ZIdx);

                int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

                gsl_complex h_i = scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx)];
                gsl_complex f_i = form_factors[calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx)];

                data_file << eps << "," << X << "," << Z << ","
                          << GSL_REAL(h_i) << (GSL_IMAG(h_i) < 0 ? "-" : "+") << abs(GSL_IMAG(h_i)) << "i" << ","
                          << GSL_REAL(f_i) << (GSL_IMAG(f_i) < 0 ? "-" : "+") << abs(GSL_IMAG(f_i)) << "i" << ","
                          << calcSquaredNormOfScatteringMatrix(contour_def_idx, externalImpulseIdx) << std::endl;
            }
        }
    }
}

int ScatteringProcess::calcScatteringAmpIdx(int basisElemIdx, int contour_def_idx, int externalImpulseIdx)
{
    return basisElemIdx * len_contour_def_epsilon * externalImpulseGrid.getLength() + contour_def_idx * externalImpulseGrid.getLength() + externalImpulseIdx;
}

double ScatteringProcess::calcSquaredNormOfScatteringMatrix(int contour_def_idx, int externalImpulseIdx)
{
    double squared_scattering_matrix_elem = scattering_amplitude[contour_def_idx * externalImpulseGrid.getLength() + externalImpulseIdx].absSquare();
    return squared_scattering_matrix_elem;
}

void ScatteringProcess::build_h_vector(int contour_def_idx, int externalImpulseIdx, gsl_vector_complex* h)
{
    gsl_vector_complex_set_zero(h);

    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
    {
        int scatteringAmpIdx = calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx);
        gsl_vector_complex_set(h, basisElemIdx, scattering_amplitude_basis_projected[scatteringAmpIdx]);
    }
}

void ScatteringProcess::calculateFormFactors(int contour_def_idx, int XIdx, int ZIdx, gsl_vector_complex* f)
{
    int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

    gsl_vector_complex* h = gsl_vector_complex_alloc(tensorBasis.getTensorBasisElementCount());
    build_h_vector(contour_def_idx, externalImpulseIdx, h);

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

    for(int contour_def_idx = 0; contour_def_idx < len_contour_def_epsilon; contour_def_idx++)
    {
        for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
        {
            for (int ZIdx = 0; ZIdx < externalImpulseGrid.getLenZ(); ZIdx++)
            {
                int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, ZIdx);

                // find f
                gsl_vector_complex_set_zero(f);
                calculateFormFactors(contour_def_idx, XIdx, ZIdx, f);

                // loop over tensor basis
                scattering_amplitude[contour_def_idx * externalImpulseGrid.getLength() + externalImpulseIdx].setZero();
                for (int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
                {
                    form_factors[calcScatteringAmpIdx(basisElemIdx, contour_def_idx, externalImpulseIdx)] = gsl_vector_complex_get(f, basisElemIdx);

                    scattering_amplitude[contour_def_idx * externalImpulseGrid.getLength() + externalImpulseIdx] += (*tensorBasis.basisTensor(basisElemIdx, externalImpulseIdx)) * gsl_vector_complex_get(f, basisElemIdx);
                }
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
