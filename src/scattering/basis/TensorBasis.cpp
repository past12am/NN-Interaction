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
#include "../../../include/utils/print/PrintGSLElements.hpp"

void TensorBasis::calculateBasis(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent,
                                 gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f,
                                 gsl_vector_complex* k_i, gsl_vector_complex* P)
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
    gsl_matrix_complex* P_Slash = gsl_matrix_complex_alloc(4, 4);
    DiracStructuresHelper::diracStructures.slash(P, P_Slash);

    //gsl_matrix_complex* KSlash = gsl_matrix_complex_alloc(4, 4);
    //DiracStructuresHelper::diracStructures.slash(K, KSlash);


    // temp variables
    gsl_matrix_complex* tmp1 = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp2 = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp3 = gsl_matrix_complex_alloc(4, 4);

    gsl_matrix_complex* tmp_down = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp_up = gsl_matrix_complex_alloc(4, 4);

    gsl_matrix_complex_set_zero(tmp1);
    gsl_matrix_complex_set_zero(tmp2);
    gsl_matrix_complex_set_zero(tmp3);
    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);



    // Tensor Basis
    // tau[0] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).1.Lambda(k_i)
    matProd3Elem(Lambda_pf, Projectors::getUnitM(), Lambda_pi, tmp1, tmp_down);
    matProd3Elem(Lambda_kf, Projectors::getUnitM(), Lambda_ki, tmp1, tmp_up);
    tauGridCurrent[0][impulseIdx] = Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);



    // tau[1] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).Slash(P).Lambda(k_i)
    matProd3Elem(Lambda_pf, Projectors::getUnitM(), Lambda_pi, tmp1, tmp_down);
    matProd3Elem(Lambda_kf, P_Slash, Lambda_ki, tmp1, tmp_up);
    tauGridCurrent[1][impulseIdx] = Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);



    // tau[2] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma[mu].Lambda(k_i)
    tauGridCurrent[2][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_pi, tmp1, tmp_down);
        matProd3Elem(Lambda_kf, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_ki, tmp1, tmp_up);

        tauGridCurrent[2][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);
    }

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);





    // tau[3] = Sum(mu = 1 to 4) Lambda(p_f).gamma[mu].Lambda(p_i) (x) Lambda(k_f).[gamma[mu], Slash(P)].Lambda(k_i)
    tauGridCurrent[3][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma[mu], Lambda_pi, tmp1, tmp_down);

        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], P_Slash, tmp2);
        matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);

        tauGridCurrent[3][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);


        gsl_matrix_complex_set_zero(tmp2);
    }

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);





    // tau[4] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Lambda(k_i)
    matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma5, Lambda_pi, tmp1, tmp_down);
    matProd3Elem(Lambda_kf, DiracStructuresHelper::diracStructures.gamma5, Lambda_ki, tmp1, tmp_up);
    tauGridCurrent[4][impulseIdx] = Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);





    // tau[5] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Slash(P).Lambda(k_i)
    matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma5, Lambda_pi, tmp1, tmp_down);

    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, P_Slash,
                   gsl_complex_rect(0, 0), tmp2);
    matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);

    tauGridCurrent[5][impulseIdx] = Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);


    gsl_matrix_complex_set_zero(tmp2);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);






    // tau[6] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.gamma[mu].Lambda(k_i)
    tauGridCurrent[6][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, DiracStructuresHelper::diracStructures.gamma[mu],
                       gsl_complex_rect(0, 0), tmp2);

        matProd3Elem(Lambda_pf, tmp2, Lambda_pi, tmp1, tmp_down);
        matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);

        tauGridCurrent[6][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);


        gsl_matrix_complex_set_zero(tmp2);
    }

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);






    // tau[7] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.[gamma[mu], Slash(P)].Lambda(k_i)
    tauGridCurrent[7][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, DiracStructuresHelper::diracStructures.gamma[mu],
                       gsl_complex_rect(0, 0), tmp2);

        matProd3Elem(Lambda_pf, tmp2, Lambda_pi, tmp1, tmp_down);
        gsl_matrix_complex_set_zero(tmp2);


        Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu], P_Slash, tmp3);
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), DiracStructuresHelper::diracStructures.gamma5, tmp3,
                       gsl_complex_rect(0, 0), tmp2);
        matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);
        gsl_matrix_complex_set_zero(tmp2);


        tauGridCurrent[7][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);
    }




    // Free temporary vars
    gsl_matrix_complex_free(tmp1);
    gsl_matrix_complex_free(tmp2);
    gsl_matrix_complex_free(tmp3);

    gsl_matrix_complex_free(tmp_down);
    gsl_matrix_complex_free(tmp_up);

    // Free dirac structures
    //gsl_matrix_complex_free(KSlash);
    gsl_matrix_complex_free(P_Slash);

    // Free projectors
    gsl_matrix_complex_free(Lambda_ki);
    gsl_matrix_complex_free(Lambda_kf);
    gsl_matrix_complex_free(Lambda_pi);
    gsl_matrix_complex_free(Lambda_pf);
}

void TensorBasis::matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C, gsl_matrix_complex* tmp, gsl_matrix_complex* res)
{
    gsl_matrix_complex_set_zero(tmp);
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
    tauGridTimelike = new Tensor4<4, 4, 4, 4>* [8];

    KMatrixGrid = new gsl_matrix_complex*[len];
    KInverseMatrixGrid = new gsl_matrix_complex*[len];
    for(int i = 0; i < 8; i++)
    {
        tauGrid[i] = new Tensor4<4, 4, 4, 4>[len];
        tauGridTimelike[i] = new Tensor4<4, 4, 4, 4>[len];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        KMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(8, 8);
        KInverseMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(8, 8);

        calculateBasis(impulseIdx, tauGrid, externalImpulseGrid->get_p_f(impulseIdx), externalImpulseGrid->get_p_i(impulseIdx),
                       externalImpulseGrid->get_k_f(impulseIdx),
                       externalImpulseGrid->get_k_i(impulseIdx), externalImpulseGrid->get_P(impulseIdx));

        //calculateBasis(impulseIdx, tauGrid, externalImpulseGrid->get_p_f_timelike(impulseIdx), externalImpulseGrid->get_p_i_timelike(impulseIdx),
        //               externalImpulseGrid->get_k_f_timelike(impulseIdx),
        //               externalImpulseGrid->get_k_i_timelike(impulseIdx), externalImpulseGrid->get_P_timelike(impulseIdx));


        calculateBasis(impulseIdx, tauGridTimelike, externalImpulseGrid->get_p_f_timelike(impulseIdx), externalImpulseGrid->get_p_i_timelike(impulseIdx),
                       externalImpulseGrid->get_k_f_timelike(impulseIdx),
                       externalImpulseGrid->get_k_i_timelike(impulseIdx), externalImpulseGrid->get_P_timelike(impulseIdx));

        calculateKMatrix(impulseIdx, tauGrid);
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
        delete []tauGridTimelike[i];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        gsl_matrix_complex_free(KMatrixGrid[impulseIdx]);
        gsl_matrix_complex_free(KInverseMatrixGrid[impulseIdx]);
    }

    delete []tauGrid;
    delete []tauGridTimelike;
    delete []KMatrixGrid;
    delete []KInverseMatrixGrid;
}

int TensorBasis::getTensorBasisElementCount() const
{
    return 8;
}

void TensorBasis::calculateKMatrix(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent)
{
    for(int i = 0; i < getTensorBasisElementCount(); i++)
    {
        for(int j = 0; j < getTensorBasisElementCount(); j++)
        {
            gsl_matrix_complex_set(KMatrixGrid[impulseIdx], j, i,
                                   tauGridCurrent[i][impulseIdx].leftContractWith(&tauGridCurrent[j][impulseIdx]));
        }
    }

    //std::cout << "K-Matrix: " << std::endl << PrintGSLElements::print_gsl_matrix_structure(KMatrixGrid[impulseIdx], 1E-10) << std::endl;
    //std::cout << "K-Matrix: " << std::endl << PrintGSLElements::print_gsl_matrix_complex(KMatrixGrid[impulseIdx]) << std::endl << std::endl;
}

void TensorBasis::calculateKMatrixInverse(int impulseIdx)
{
    int signum;
    gsl_permutation* p = gsl_permutation_alloc(8);

    gsl_matrix_complex* LUDecomp = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_memcpy(LUDecomp, KMatrixGrid[impulseIdx]);

    gsl_linalg_complex_LU_decomp(LUDecomp, p, &signum);
    // TODO check why K gets singulary
    gsl_complex det = gsl_linalg_complex_LU_det(LUDecomp, signum);

    // Calc Determinant Manually
    //gsl_complex det_manually = gsl_complex_rect(0, 0);
    //for (int i = 1; i < 8; i++)
    //    det_manually = gsl_complex_mul(det_manually, gsl_matrix_complex_get(LUDecomp, i, i));
    //det_manually = gsl_complex_mul_real(det_manually, pow(-1, signum));

    if(gsl_isnan(GSL_REAL(det)) || gsl_isnan(GSL_IMAG(det)) || gsl_complex_abs(det) == 0)
    {
        std::cout << "Singular Basis inversion matrix at impulseIdx " << impulseIdx << std::endl;
    }
    else
    {
        gsl_linalg_complex_LU_invert(LUDecomp, p, KInverseMatrixGrid[impulseIdx]);
    }

    std::cout << "K-Matrix Inverse: " << std::endl << PrintGSLElements::print_gsl_matrix_structure(KInverseMatrixGrid[impulseIdx], 1E-10) << std::endl;

}

gsl_matrix_complex* TensorBasis::K(int impulseIdx)
{
    return KMatrixGrid[impulseIdx];
}

gsl_matrix_complex* TensorBasis::KInv(int impulseIdx)
{
    return KInverseMatrixGrid[impulseIdx];
}




