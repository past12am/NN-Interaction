//
// Created by past12am on 8/3/23.
//

#include "../../../include/scattering/basis/TensorBasis.hpp"

#include "complex"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_complex_math.h"
#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/utils/math/Commutator.hpp"

void TensorBasis::calculateBasis(int impulseIdx,
                                 gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f,
                                 gsl_vector_complex* k_i, gsl_vector_complex* P, gsl_vector_complex* K)
{
    // Positive Energy Projectors
    gsl_matrix_complex* Lambda_pf = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(p_f, Lambda_pf);

    gsl_matrix_complex* Lambda_pi = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(p_i, Lambda_pi);

    gsl_matrix_complex* Lambda_kf = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(k_f, Lambda_kf);

    gsl_matrix_complex* Lambda_ki = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(k_i, Lambda_ki);


    // Dirac Structures
    gsl_matrix_complex* PSlash = gsl_matrix_complex_alloc(4, 4);
    DiracStructuresHelper::diracStructures.slash(P, PSlash);

    gsl_matrix_complex* KSlash = gsl_matrix_complex_alloc(4, 4);
    DiracStructuresHelper::diracStructures.slash(K, KSlash);


    // temp variables
    gsl_matrix_complex* tmp = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpA = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpB = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpC = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpSumA = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpSumB = gsl_matrix_complex_alloc(4, 4);



    // Tensor Basis
    // tau[0] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).1.Lambda(k_i)
    matProd3Elem(Lambda_pf, Projectors::getUnitM(), Lambda_pi, tmp, tmpA);
    matProd3Elem(Lambda_kf, Projectors::getUnitM(), Lambda_ki, tmp, tmpB);
    tauGrid[impulseIdx][0] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[1] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).Slash(P).Lambda(k_i)
    // keep tmpA (same as tau[0])
    matProd3Elem(Lambda_kf, PSlash, Lambda_ki, tmpC, tmpB);
    tauGrid[impulseIdx][1] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[2] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma[mu].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumA);
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_pi, tmp, tmpA);
        matProd3Elem(Lambda_kf, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_ki, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[impulseIdx][2] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[3] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).[gamma[mu], Slash(P)].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        // tmpA same as for tau[2]
        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], PSlash, tmpC);
        matProd3Elem(Lambda_kf, tmpC, Lambda_ki, tmp, tmpB);

        //gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[impulseIdx][3] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[4] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Lambda(k_i)
    matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma5, Lambda_pi, tmp, tmpA);
    matProd3Elem(Lambda_kf, DiracStructuresHelper::diracStructures.gamma5, Lambda_ki, tmp, tmpB);
    tauGrid[impulseIdx][4] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[5] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Slash(P).Lambda(k_i)
    // keep tmpA (same as tau[4])
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, PSlash,
                   gsl_complex_rect(0, 0), tmpC);
    matProd3Elem(Lambda_kf, tmpC, Lambda_ki, tmp, tmpB);
    tauGrid[impulseIdx][5] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[6] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.gamma[mu].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumA);
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, DiracStructuresHelper::diracStructures.gamma[mu],
                       gsl_complex_rect(0, 0), tmpC);

        matProd3Elem(Lambda_pf, tmpC, Lambda_pi, tmp, tmpA);
        matProd3Elem(Lambda_kf, tmpC, Lambda_ki, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[impulseIdx][6] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[7] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.[gamma[mu], Slash(P)].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        // using tmpA as working register for tmpB as we already calculated tmpSumA
        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], PSlash, tmpA);
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, tmpA,
                       gsl_complex_rect(0, 0), tmpC);
        matProd3Elem(Lambda_kf, tmpC, Lambda_ki, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[impulseIdx][7] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // Free temporary vars
    gsl_matrix_complex_free(tmpSumB);
    gsl_matrix_complex_free(tmpSumA);
    gsl_matrix_complex_free(tmpC);
    gsl_matrix_complex_free(tmpB);
    gsl_matrix_complex_free(tmpA);
    gsl_matrix_complex_free(tmp);

    // Free dirac structures
    gsl_matrix_complex_free(KSlash);
    gsl_matrix_complex_free(PSlash);

    // Free projectors
    gsl_matrix_complex_free(Lambda_ki);
    gsl_matrix_complex_free(Lambda_kf);
    gsl_matrix_complex_free(Lambda_pi);
    gsl_matrix_complex_free(Lambda_pf);
}

void TensorBasis::matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C, gsl_matrix_complex* tmp, gsl_matrix_complex* res)
{
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), A, B,
                   gsl_complex_rect(0, 0), tmp);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), tmp, C,
                   gsl_complex_rect(0, 0), res);
}

TensorBasis::operator std::string() const
{
    // TODO
    return std::string();
}

TensorBasis::TensorBasis(ExternalImpulseGrid* externalImpulseGrid) : len(externalImpulseGrid->getLength()), externalImpulseGrid(externalImpulseGrid)
{
    tauGrid = new Tensor4<4, 4, 4, 4>* [len];
    for(int i = 0; i < len; i++)
    {
        tauGrid[i] = new Tensor4<4, 4, 4, 4>[8];
        calculateBasis(i, externalImpulseGrid->get_p_f(i), externalImpulseGrid->get_p_i(i), externalImpulseGrid->get_k_f(i),
                       externalImpulseGrid->get_k_i(i), externalImpulseGrid->get_P(i), externalImpulseGrid->get_K(i));
    }
}

Tensor4<4, 4, 4, 4>* TensorBasis::tau(int impulseIdx)
{
    return tauGrid[impulseIdx];
}

TensorBasis::~TensorBasis()
{
    for(int i = 0; i < len; i++)
    {
        delete tauGrid[i];
    }
}




