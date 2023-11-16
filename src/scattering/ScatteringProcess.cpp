//
// Created by past12am on 8/18/23.
//

#include <fstream>
#include <gsl/gsl_blas.h>
#include <cassert>
#include <iostream>
#include "../../include/scattering/ScatteringProcess.hpp"
#include "../../include/Definitions.h"


ScatteringProcess::ScatteringProcess(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, double a, int threadIdx) :
        nucleon_mass(nucleon_mass), threadIdx(threadIdx),
        externalImpulseGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper, nucleon_mass, a),
        tensorBasis(&externalImpulseGrid, nucleon_mass)
{
    scattering_amplitude_basis_projected = new gsl_complex[tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength()];
    form_factors = new gsl_complex[tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength()];

    inverseKMatrix = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_set_zero(inverseKMatrix);

    scattering_matrix = new Tensor4<4, 4, 4, 4>[externalImpulseGrid.getLength()];
    for(int i = 0; i < externalImpulseGrid.getLength(); i++)
    {
        scattering_matrix[i] = Tensor4<4, 4, 4, 4>();
    }

    l = gsl_vector_complex_alloc(4);
}

ScatteringProcess::~ScatteringProcess()
{
    gsl_vector_complex_free(l);

    delete[] scattering_matrix;

    gsl_matrix_complex_free(inverseKMatrix);

    delete[] form_factors;
    delete[] scattering_amplitude_basis_projected;
}

void ScatteringProcess::calc_l(double l2, double z, double y, double phi, gsl_vector_complex* l)
{
    gsl_vector_complex_set(l, 0, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(l, 1, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * y, 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(z, 0));

    gsl_vector_complex_scale(l, gsl_complex_rect(sqrt(l2), 0));
}

gsl_complex ScatteringProcess::integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double l2, double z, double y, double phi)
{
    //gsl_vector_complex* l = gsl_vector_complex_alloc(4);
    if(!l_mutex.try_lock())
    {
        std::cout << "Probable race condition on temporary l impulse" << std::endl;
        exit(-1);
    }

    gsl_vector_complex_set_zero(l);

    calc_l(l2, z, y, phi, l);

    // get basis Element
    Tensor4<4, 4, 4, 4>* tau_current = tensorBasis.tau(basisElemIdx, externalImpulseIdx);

    // get Tensor
    Tensor4<4, 4, 4, 4> integralKernelTensor;
    integralKernel(l,
                   externalImpulseGrid.get_Q(externalImpulseIdx), externalImpulseGrid.get_K(externalImpulseIdx), externalImpulseGrid.get_P(externalImpulseIdx),
                   externalImpulseGrid.get_p_f(externalImpulseIdx), externalImpulseGrid.get_p_i(externalImpulseIdx),
                   externalImpulseGrid.get_k_f(externalImpulseIdx), externalImpulseGrid.get_k_i(externalImpulseIdx),
                   &integralKernelTensor);

    l_mutex.unlock();

    gsl_complex kernel_res = integralKernelTensor.leftContractWith(tau_current);
    return kernel_res;
}

void ScatteringProcess::store_scattering_amplitude(int basisElemIdx, double a, std::ofstream& data_file)
{
    for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
    {
        double X = externalImpulseGrid.calcXAt(XIdx);

        for(int zIdx = 0; zIdx < externalImpulseGrid.getLenZ(); zIdx++)
        {
            double z = externalImpulseGrid.calcZAt(zIdx);

            int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, zIdx);
            gsl_complex PK;
            gsl_complex QQ;

            gsl_blas_zdotu(externalImpulseGrid.get_P(externalImpulseIdx), externalImpulseGrid.get_K(externalImpulseIdx), &PK);
            gsl_blas_zdotu(externalImpulseGrid.get_Q(externalImpulseIdx), externalImpulseGrid.get_Q(externalImpulseIdx), &QQ);

            assert(GSL_IMAG(PK) == 0);
            assert(GSL_IMAG(QQ) == 0);

            gsl_complex h_i = scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)];
            gsl_complex f_i = form_factors[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)];

            data_file << a << "," << X << "," << z << "," << GSL_REAL(PK) << "," << GSL_REAL(QQ) << ","
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
    double squared_scattering_matrix_elem = scattering_matrix[externalImpulseIdx].absSquare();
    return squared_scattering_matrix_elem;
}


gsl_matrix_complex* ScatteringProcess::buildInverseK(double tau, double z, gsl_complex M)
{
    double pref_00 = (pow(z, 2) * (-1.0 + tau) - tau)/((-1.0 + pow(z, 2)) * tau * pow(1.0 + tau, 4));
    gsl_complex elem_00 = gsl_complex_rect(pref_00, 0);
    gsl_matrix_complex_set(inverseKMatrix, 0, 0, elem_00);


    double pref_01 = -z / ((-1.0 + pow(z, 2)) * tau * pow(1.0 + tau, 4));
    gsl_complex  elem_01 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_01);
    gsl_matrix_complex_set(inverseKMatrix, 0, 1, elem_01);
    gsl_matrix_complex_set(inverseKMatrix, 1, 0, elem_01);

    double pref_11 = -(2 + 3 * tau + pow(tau, 2) - pow(z, 2) * (2.0 + tau + pow(tau, 2)))/(pow((-1.0 + pow(z, 2)), 2) * tau * pow(1.0 + tau, 6));
    gsl_complex elem_11 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(1, 0), gsl_complex_pow_real(M, 2)), pref_11);
    gsl_matrix_complex_set(inverseKMatrix, 1, 1, elem_11);

    double pref_12 = ((1.0 + pow(z, 2) * (-1.0 + tau) + tau)) / (pow(-1.0 + pow(z, 2), 2) * tau * pow(1.0 + tau, 5));
    gsl_complex elem_12 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_12);
    gsl_matrix_complex_set(inverseKMatrix, 1, 2, elem_12);
    gsl_matrix_complex_set(inverseKMatrix, 2, 1, elem_12);

    double pref_14 = -(2.0 * z) / ((-1 + pow(z, 2)) * tau * pow(1 + tau, 5));
    gsl_complex elem_14 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_14);
    gsl_matrix_complex_set(inverseKMatrix, 1, 4, elem_14);
    gsl_matrix_complex_set(inverseKMatrix, 4, 1, elem_14);

    double pref_16 = -(2.0 * z) / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 5));
    gsl_complex elem_16 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_16);
    gsl_matrix_complex_set(inverseKMatrix, 1, 6, elem_16);
    gsl_matrix_complex_set(inverseKMatrix, 6, 1, elem_16);

    double pref_17 = pow(z, 2) / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 6));
    gsl_complex elem_17 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(1, 0), gsl_complex_pow_real(M, 2)), pref_17);
    gsl_matrix_complex_set(inverseKMatrix, 1, 7, elem_17);
    gsl_matrix_complex_set(inverseKMatrix, 7, 1, elem_17);


    double pref_22 = (1 + pow(z, 2) * (-1 + tau) + tau) / (pow(-1 + pow(z, 2), 2) * tau * pow(1 + tau, 4));
    gsl_complex elem_22 = gsl_complex_rect(pref_22, 0);
    gsl_matrix_complex_set(inverseKMatrix, 2, 2, elem_22);

    double pref_24 = -(2 * z) / ((-1 + pow(z, 2)) * tau * pow(1 + tau, 4));
    gsl_complex elem_24 = gsl_complex_rect(pref_24, 0);
    gsl_matrix_complex_set(inverseKMatrix, 2, 4, elem_24);
    gsl_matrix_complex_set(inverseKMatrix, 4, 2, elem_24);

    double pref_26 = -(2 * z) / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 4));
    gsl_complex elem_26 = gsl_complex_rect(pref_26, 0);
    gsl_matrix_complex_set(inverseKMatrix, 2, 6, elem_26);
    gsl_matrix_complex_set(inverseKMatrix, 6, 2, elem_26);

    double pref_27 = -pow(z, 2) / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 5));
    gsl_complex elem_27 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_27);
    gsl_matrix_complex_set(inverseKMatrix, 2, 7, elem_27);
    gsl_matrix_complex_set(inverseKMatrix, 7, 2, elem_27);


    double pref_33 = -1.0/(4.0 * (-1 + pow(z, 2)) * tau * pow(1 + tau, 4));
    gsl_complex elem_33 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(1, 0), gsl_complex_pow_real(M, 2)), pref_33);
    gsl_matrix_complex_set(inverseKMatrix, 3, 3, elem_33);


    double pref_44 = -(2 + 2 * tau + pow(tau, 2) - pow(z, 2) * (2 + tau + pow(tau, 2))) / ((-1 + pow(z, 2)) * pow(tau, 2) * pow(1 + tau, 4));
    gsl_complex elem_44 = gsl_complex_rect(pref_44, 0);
    gsl_matrix_complex_set(inverseKMatrix, 4, 4, elem_44);

    double pref_46 = 1.0 / ((-1 + pow(z, 2)) * tau * pow(1 + tau, 4));
    gsl_complex elem_46 = gsl_complex_rect(pref_46, 0);
    gsl_matrix_complex_set(inverseKMatrix, 4, 6, elem_46);
    gsl_matrix_complex_set(inverseKMatrix, 6, 4, elem_46);

    double pref_47 = -(z * (-1 + tau)) / (2 * (-1 + pow(z, 2)) * tau * pow(1 + tau, 5));
    gsl_complex elem_47 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_47);
    gsl_matrix_complex_set(inverseKMatrix, 4, 7, elem_47);
    gsl_matrix_complex_set(inverseKMatrix, 7, 4, elem_47);


    double pref_55 = -1.0 / ((-1 + pow(z, 2)) * tau * pow(1 + tau, 4));
    gsl_complex elem_55 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(1, 0), gsl_complex_pow_real(M, 2)), pref_55);
    gsl_matrix_complex_set(inverseKMatrix, 5, 5, elem_55);


    double pref_66 = (1 + pow(z, 2)) / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 4));
    gsl_complex elem_66 = gsl_complex_rect(pref_66, 0);
    gsl_matrix_complex_set(inverseKMatrix, 6, 6, elem_66);

    double pref_67 = z / (pow(-1 + pow(z, 2), 2) * pow(1 + tau, 5));
    gsl_complex elem_67 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(0, 1), M), pref_67);
    gsl_matrix_complex_set(inverseKMatrix, 6, 7, elem_67);
    gsl_matrix_complex_set(inverseKMatrix, 7, 6, elem_67);


    double pref_77 = -(1 - pow(z, 2) * (-1 + tau) + tau) / (4.0 * pow(-1 + pow(z, 2), 2) * pow(1 + tau, 6));
    gsl_complex elem_77 = gsl_complex_mul_real(gsl_complex_div(gsl_complex_rect(1, 0), gsl_complex_pow_real(M, 2)), pref_77);
    gsl_matrix_complex_set(inverseKMatrix, 7, 7, elem_77);


    return inverseKMatrix;
}

void ScatteringProcess::build_h_vector(int externalImpulseIdx, gsl_vector_complex* h)
{
    gsl_vector_complex_set_zero(h);

    for(int basisElemIdx = 0; basisElemIdx < 8; basisElemIdx++)
    {
        int scatteringAmpIdx = calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx);
        gsl_vector_complex_set(h, basisElemIdx, scattering_amplitude_basis_projected[scatteringAmpIdx]);
    }
}

void ScatteringProcess::calculateFormFactors(int XIdx, int zIdx, gsl_complex M, gsl_vector_complex* f)
{
    int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, zIdx);

    gsl_vector_complex* h = gsl_vector_complex_alloc(8);
    build_h_vector(externalImpulseIdx, h);

    gsl_matrix_complex* invK = tensorBasis.KInv(externalImpulseIdx);

    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, invK, h, GSL_COMPLEX_ZERO, f);
    gsl_vector_complex_free(h);
}

void ScatteringProcess::buildScatteringMatrix()
{
    gsl_vector_complex* f = gsl_vector_complex_alloc(8);

    for(int XIdx = 0; XIdx < externalImpulseGrid.getLenX(); XIdx++)
    {
        for (int zIdx = 0; zIdx < externalImpulseGrid.getLenZ(); zIdx++)
        {
            int externalImpulseIdx = externalImpulseGrid.getGridIdx(XIdx, zIdx);

            // find f
            calculateFormFactors(XIdx, zIdx, nucleon_mass, f);

            // loop over tensor basis
            for (int basisElemIdx = 0; basisElemIdx < 8; basisElemIdx++)
            {
                //assert(GSL_IMAG(gsl_vector_complex_get(f, basisElemIdx)) == 0);
                form_factors[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)] = gsl_vector_complex_get(f, basisElemIdx);
                scattering_matrix[externalImpulseIdx] += (*tensorBasis.tau(basisElemIdx, externalImpulseIdx)) * gsl_vector_complex_get(f, basisElemIdx);

                std::cout << "At basis " << basisElemIdx << ": " << scattering_matrix[externalImpulseIdx] << std::endl;
            }
        }
    }
}

void ScatteringProcess::performScatteringCalculation(double l2_cutoff)
{
    integrate(l2_cutoff);
    buildScatteringMatrix();
}

TensorBasis* ScatteringProcess::getTensorBasis()
{
    return &tensorBasis;
}
