//
// Created by past12am on 8/3/23.
//

#include "../../../include/scattering/basis/TensorBasis.hpp"

#include "complex"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_linalg.h"
#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/utils/math/Commutator.hpp"

void TensorBasis::calculateBasis(int impulseIdx,
                                 gsl_vector_complex* p_f_timelike, gsl_vector_complex* p_i_timelike, gsl_vector_complex* k_f_timelike,
                                 gsl_vector_complex* k_i_timelike, gsl_vector_complex* P_timelike)
{
    // Positive Energy Projectors
    gsl_matrix_complex* Lambda_pf_timelike = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(p_f_timelike, Lambda_pf_timelike);

    gsl_matrix_complex* Lambda_pi_timelike = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(p_i_timelike, Lambda_pi_timelike);

    gsl_matrix_complex* Lambda_kf_timelike = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(k_f_timelike, Lambda_kf_timelike);

    gsl_matrix_complex* Lambda_ki_timelike = gsl_matrix_complex_alloc(4, 4);
    Projectors::posEnergyProjector(k_i_timelike, Lambda_ki_timelike);


    // Dirac Structures
    gsl_matrix_complex* P_timelike_Slash = gsl_matrix_complex_alloc(4, 4);
    DiracStructuresHelper::diracStructures.slash(P_timelike, P_timelike_Slash);

    //gsl_matrix_complex* KSlash = gsl_matrix_complex_alloc(4, 4);
    //DiracStructuresHelper::diracStructures.slash(K, KSlash);


    // temp variables
    gsl_matrix_complex* tmp = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpA = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpB = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpC = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpSumA = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmpSumB = gsl_matrix_complex_alloc(4, 4);



    // Tensor Basis
    // tau[0] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).1.Lambda(k_i)
    matProd3Elem(Lambda_pf_timelike, Projectors::getUnitM(), Lambda_pi_timelike, tmp, tmpA);
    matProd3Elem(Lambda_kf_timelike, Projectors::getUnitM(), Lambda_ki_timelike, tmp, tmpB);
    tauGrid[0][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[1] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).Slash(P).Lambda(k_i)
    // keep tmpA (same as tau[0])
    matProd3Elem(Lambda_kf_timelike, P_timelike_Slash, Lambda_ki_timelike, tmpC, tmpB);
    tauGrid[1][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[2] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma[mu].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumA);
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        matProd3Elem(Lambda_pf_timelike, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_pi_timelike, tmp, tmpA);
        matProd3Elem(Lambda_kf_timelike, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_ki_timelike, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[2][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[3] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).[gamma[mu], Slash(P)].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        // tmpA same as for tau[2]
        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], P_timelike_Slash, tmpC);
        matProd3Elem(Lambda_kf_timelike, tmpC, Lambda_ki_timelike, tmp, tmpB);

        //gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[3][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[4] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Lambda(k_i)
    matProd3Elem(Lambda_pf_timelike, DiracStructuresHelper::diracStructures.gamma5, Lambda_pi_timelike, tmp, tmpA);
    matProd3Elem(Lambda_kf_timelike, DiracStructuresHelper::diracStructures.gamma5, Lambda_ki_timelike, tmp, tmpB);
    tauGrid[4][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[5] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Slash(P).Lambda(k_i)
    // keep tmpA (same as tau[4])
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, P_timelike_Slash,
                   gsl_complex_rect(0, 0), tmpC);
    matProd3Elem(Lambda_kf_timelike, tmpC, Lambda_ki_timelike, tmp, tmpB);
    tauGrid[5][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpA, tmpB);


    // tau[6] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.gamma[mu].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumA);
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, DiracStructuresHelper::diracStructures.gamma[mu],
                       gsl_complex_rect(0, 0), tmpC);

        matProd3Elem(Lambda_pf_timelike, tmpC, Lambda_pi_timelike, tmp, tmpA);
        matProd3Elem(Lambda_kf_timelike, tmpC, Lambda_ki_timelike, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumA, tmpA);
        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[6][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // tau[7] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.[gamma[mu], Slash(P)].Lambda(k_i)
    gsl_matrix_complex_set_zero(tmpSumB);
    for(int mu = 0; mu < 4; mu++)
    {
        // using tmpA as working register for tmpB as we already calculated tmpSumA
        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], P_timelike_Slash, tmpA);
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, tmpA,
                       gsl_complex_rect(0, 0), tmpC);
        matProd3Elem(Lambda_kf_timelike, tmpC, Lambda_ki_timelike, tmp, tmpB);

        gsl_matrix_complex_add(tmpSumB, tmpB);
    }
    tauGrid[7][impulseIdx] = Tensor4<4, 4, 4, 4>(tmpSumA, tmpSumB);


    // Free temporary vars
    gsl_matrix_complex_free(tmpSumB);
    gsl_matrix_complex_free(tmpSumA);
    gsl_matrix_complex_free(tmpC);
    gsl_matrix_complex_free(tmpB);
    gsl_matrix_complex_free(tmpA);
    gsl_matrix_complex_free(tmp);

    // Free dirac structures
    //gsl_matrix_complex_free(KSlash);
    gsl_matrix_complex_free(P_timelike_Slash);

    // Free projectors
    gsl_matrix_complex_free(Lambda_ki_timelike);
    gsl_matrix_complex_free(Lambda_kf_timelike);
    gsl_matrix_complex_free(Lambda_pi_timelike);
    gsl_matrix_complex_free(Lambda_pf_timelike);
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

TensorBasis::TensorBasis(ExternalImpulseGrid* externalImpulseGrid) : len(externalImpulseGrid->getLength())
{
    tauGrid = new Tensor4<4, 4, 4, 4>* [8];
    KMatrixGrid = new gsl_matrix_complex*[len];
    KInverseMatrixGrid = new gsl_matrix_complex*[len];
    for(int i = 0; i < 8; i++)
    {
        tauGrid[i] = new Tensor4<4, 4, 4, 4>[len];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        KMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(8, 8);
        KInverseMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(8, 8);

        calculateBasis(impulseIdx, externalImpulseGrid->get_p_f_timelike(impulseIdx), externalImpulseGrid->get_p_i_timelike(impulseIdx),
                       externalImpulseGrid->get_k_f_timelike(impulseIdx),
                       externalImpulseGrid->get_k_i_timelike(impulseIdx), externalImpulseGrid->get_P_timelike(impulseIdx));

        calculateKMatrix(impulseIdx);
        calculateKMatrixInverse(impulseIdx);
    }
}

Tensor4<4, 4, 4, 4>* TensorBasis::tauGridAt(int basisElemIdx)
{
    return tauGrid[basisElemIdx];
}

Tensor4<4, 4, 4, 4>* TensorBasis::tau(int basisElemIdx, int externalImpulseIdx)
{
    return &(tauGrid[basisElemIdx][externalImpulseIdx]);
}

TensorBasis::~TensorBasis()
{
    for(int i = 0; i < 8; i++)
    {
        delete []tauGrid[i];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        gsl_matrix_complex_free(KMatrixGrid[impulseIdx]);
        gsl_matrix_complex_free(KInverseMatrixGrid[impulseIdx]);
    }

    delete []tauGrid;
    delete []KMatrixGrid;
    delete []KInverseMatrixGrid;
}

int TensorBasis::getTensorBasisElementCount() const
{
    return 8;
}

void TensorBasis::calculateKMatrix(int impulseIdx)
{
    for(int i = 0; i < getTensorBasisElementCount(); i++)
    {
        for(int j = 0; j < getTensorBasisElementCount(); j++)
        {
            gsl_matrix_complex_set(KMatrixGrid[impulseIdx], i, j, tauGrid[i][impulseIdx].contractTauOther(tauGrid[j][impulseIdx]));
        }
    }
}

void TensorBasis::calculateKMatrixInverse(int impulseIdx)
{
    int signum;
    gsl_permutation* p = gsl_permutation_alloc(8);

    gsl_matrix_complex* LUDecomp = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_memcpy(LUDecomp, KMatrixGrid[impulseIdx]);

    gsl_linalg_complex_LU_decomp(LUDecomp, p, &signum);
    gsl_linalg_complex_LU_invert(LUDecomp, p, KInverseMatrixGrid[impulseIdx]);
}

gsl_matrix_complex* TensorBasis::K(int impulseIdx)
{
    return KMatrixGrid[impulseIdx];
}

gsl_matrix_complex* TensorBasis::KInv(int impulseIdx)
{
    return KInverseMatrixGrid[impulseIdx];
}




