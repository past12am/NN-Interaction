//
// Created by past12am on 8/18/23.
//

#include <fstream>
#include <gsl/gsl_blas.h>
#include <cassert>
#include <iostream>
#include "../../include/scattering/ScatteringProcess.hpp"


ScatteringProcess::ScatteringProcess(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex M_nucleon) : externalImpulseGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper, M_nucleon),
                                                                                                   tensorBasis(&externalImpulseGrid)
{
    scattering_amplitude_basis_projected = new gsl_complex[tensorBasis.getLength() * externalImpulseGrid.getLength()];

    inverseKMatrix = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_set_zero(inverseKMatrix);

    scattering_matrix = new Tensor4<4, 4, 4, 4>[externalImpulseGrid.getLength()];
    for(int i = 0; i < externalImpulseGrid.getLength(); i++)
    {
        scattering_matrix[i] = Tensor4<4, 4, 4, 4>();
    }
}

void ScatteringProcess::calc_l(double l2, double z, double y, double phi, gsl_vector_complex* l)
{
    gsl_vector_complex_set(l, 0, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * sin(phi), 0));
    gsl_vector_complex_set(l, 1, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * sqrt(1.0 - pow(y, 2)) * cos(phi), 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(sqrt(1.0 - pow(z, 2)) * y, 0));
    gsl_vector_complex_set(l, 2, gsl_complex_rect(z, 0));

    gsl_vector_complex_scale(l, gsl_complex_rect(sqrt(l2), 0));
}

ScatteringProcess::~ScatteringProcess()
{
    delete[] scattering_matrix;

    gsl_matrix_complex_free(inverseKMatrix);

    delete[] scattering_amplitude_basis_projected;
}

void ScatteringProcess::store_scattering_amplitude(std::string data_path)
{
    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getLength(); basisElemIdx++)
    {
        std::ostringstream fnamestrstream;
        fnamestrstream << data_path << "/tau_" << basisElemIdx << ".txt";

        std::ofstream data_file;
        data_file.open(fnamestrstream.str(), std::ofstream::out | std::ios::trunc);

        data_file << "tau,z,PK,QQ,scattering_amp,|scattering_amp|2" << std::endl;


        for(int tauIdx = 0; tauIdx < externalImpulseGrid.getLenTau(); tauIdx++)
        {
            double tau = externalImpulseGrid.calcTauAt(tauIdx);

            for(int zIdx = 0; zIdx < externalImpulseGrid.getLenZ(); zIdx++)
            {
                double z = externalImpulseGrid.calcZAt(zIdx);

                int externalImpulseIdx = externalImpulseGrid.getGridIdx(tauIdx, zIdx);
                gsl_complex PK;
                gsl_complex QQ;

                gsl_blas_zdotu(externalImpulseGrid.get_P(externalImpulseIdx), externalImpulseGrid.get_K(externalImpulseIdx), &PK);
                gsl_blas_zdotu(externalImpulseGrid.get_Q(externalImpulseIdx), externalImpulseGrid.get_Q(externalImpulseIdx), &QQ);

                assert(GSL_IMAG(PK) == 0);
                assert(GSL_IMAG(QQ) == 0);

                gsl_complex curScatteringAmpRes = scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)];

                data_file << tau << "," << z << "," << GSL_REAL(PK) << "," << GSL_REAL(QQ) << ","
                          << GSL_REAL(curScatteringAmpRes) << (GSL_IMAG(curScatteringAmpRes) < 0 ? "-" : "+") << abs(GSL_IMAG(curScatteringAmpRes)) << "i" << ","
                          << calcSquaredNormOfScatteringMatrix(externalImpulseIdx) << std::endl;
            }
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

gsl_matrix_complex* ScatteringProcess::getInverseK(double tau, double z, gsl_complex M)
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

void ScatteringProcess::calculateFormFactors(int tauIdx, int zIdx, gsl_complex M, gsl_vector_complex* f)
{
    gsl_vector_complex* h = gsl_vector_complex_alloc(8);
    build_h_vector(externalImpulseGrid.getGridIdx(tauIdx, zIdx), h);

    gsl_matrix_complex* invK = getInverseK(externalImpulseGrid.calcTauAt(tauIdx), externalImpulseGrid.calcZAt(zIdx), M);

    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, invK, h, GSL_COMPLEX_ZERO, f);
    gsl_vector_complex_free(h);
}

void ScatteringProcess::buildScatteringMatrix(gsl_complex M_nucleon)
{
    gsl_vector_complex* f = gsl_vector_complex_alloc(8);

    for(int tauIdx = 0; tauIdx < externalImpulseGrid.getLenTau(); tauIdx++)
    {
        for (int zIdx = 0; zIdx < externalImpulseGrid.getLenZ(); zIdx++)
        {
            int externalImpulseIdx = externalImpulseGrid.getGridIdx(tauIdx, zIdx);

            // find f
            calculateFormFactors(tauIdx, zIdx, M_nucleon, f);

            // loop over tensor basis
            for (int basisElemIdx = 0; basisElemIdx < 8; basisElemIdx++)
            {
                scattering_matrix[externalImpulseIdx] += (*tensorBasis.tau(basisElemIdx, externalImpulseIdx)) * gsl_vector_complex_get(f, basisElemIdx);

                std::cout << "At basis " << basisElemIdx << ": " << scattering_matrix[externalImpulseIdx] << std::endl;
            }
        }
    }
}
