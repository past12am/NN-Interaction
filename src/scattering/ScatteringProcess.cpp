//
// Created by past12am on 8/18/23.
//

#include <fstream>
#include <gsl/gsl_blas.h>
#include <cassert>
#include <iostream>
#include "../../include/scattering/ScatteringProcess.hpp"
#include "../../include/Definitions.h"


ScatteringProcess::ScatteringProcess(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, int threadIdx) :
        nucleon_mass(nucleon_mass), threadIdx(threadIdx),
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

void ScatteringProcess::calc_k(double k2, double z, double y, double phi, gsl_vector_complex* k)
{
    gsl_vector_complex_set(k, 0, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(k, 1, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(k, 2, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * y, 0));
    gsl_vector_complex_set(k, 2, gsl_complex_rect(z, 0));

    gsl_vector_complex_scale(k, gsl_complex_rect(sqrt(k2), 0));
}

gsl_complex ScatteringProcess::integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double k2, double z, double y, double phi)
{
    //gsl_vector_complex* l = gsl_vector_complex_alloc(4);
    if(!k_mutex.try_lock())
    {
        std::cout << "Probable race condition on temporary l impulse" << std::endl;
        exit(-1);
    }

    gsl_vector_complex_set_zero(k);
    calc_k(k2, z, y, phi, k);

    // get basis Element
    Tensor4<4, 4, 4, 4>* currentBasisElement = tensorBasis.tau(basisElemIdx, externalImpulseIdx);

    // get Tensor
    Tensor4<4, 4, 4, 4> integralKernelTensor = Tensor4<4, 4, 4, 4>();
    integralKernel(k,
                   externalImpulseGrid.get_l_ext(externalImpulseIdx), externalImpulseGrid.get_r_ext(externalImpulseIdx),
                   externalImpulseGrid.get_P_ext(externalImpulseIdx),
                   externalImpulseGrid.get_p_f(externalImpulseIdx), externalImpulseGrid.get_p_i(externalImpulseIdx),
                   externalImpulseGrid.get_k_f(externalImpulseIdx), externalImpulseGrid.get_k_i(externalImpulseIdx),
                   &integralKernelTensor);

    k_mutex.unlock();

    gsl_complex kernel_res = integralKernelTensor.leftContractWith(currentBasisElement);

    // Set 0 if < 1E-30
    /*
    if(abs(kernel_res.dat[0]) < 1E-30)
        kernel_res.dat[0] = 0;
    if(abs(kernel_res.dat[1]) < 1E-30)
        kernel_res.dat[1] = 0;
    */

    return kernel_res;
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

    gsl_matrix_complex* invK = tensorBasis.KInv(externalImpulseIdx);

    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, invK, h, GSL_COMPLEX_ZERO, f);
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
