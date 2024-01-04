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
#include "../../../include/Definitions.h"

void TensorBasis::calculateBasis(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent,
                                 gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f,
                                 gsl_vector_complex* k_i)
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



    // temp variables
    gsl_matrix_complex* tmp1 = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp2 = gsl_matrix_complex_alloc(4, 4);

    gsl_matrix_complex* tmp_down = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* tmp_up = gsl_matrix_complex_alloc(4, 4);

    gsl_matrix_complex_set_zero(tmp1);
    gsl_matrix_complex_set_zero(tmp2);
    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);



    // Tensor Basis
    // tau[0] = Lambda(p_f).1.Lambda(p_i) (x) Lambda(k_f).1.Lambda(k_i)
    matProd3Elem(Lambda_pf, Projectors::getUnitM(), Lambda_pi, tmp1, tmp_down);
    matProd3Elem(Lambda_kf, Projectors::getUnitM(), Lambda_ki, tmp1, tmp_up);
    tauGridCurrent[0][impulseIdx] = Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);




    // tau[1] = Lambda(p_f).gamma5.Lambda(p_i) (x) Lambda(k_f).gamma5.Lambda(k_i)
    matProd3Elem(Lambda_pf, DiracStructuresHelper::diracStructures.gamma5, Lambda_pi, tmp1, tmp_down);
    matProd3Elem(Lambda_kf, DiracStructuresHelper::diracStructures.gamma5, Lambda_ki, tmp1, tmp_up);
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




    // tau[3] = Sum(mu = 1 to 4) Lambda(p_f).gamma5.gamma[mu].Lambda(p_i) (x) Lambda(k_f).gamma5.gamma[mu].Lambda(k_i)
    tauGridCurrent[3][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        gsl_matrix_complex_set_zero(tmp2);
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0),
                       DiracStructuresHelper::diracStructures.gamma5, DiracStructuresHelper::diracStructures.gamma[mu],
                       gsl_complex_rect(0, 0), tmp2);

        matProd3Elem(Lambda_pf, tmp2, Lambda_pi, tmp1, tmp_down);
        matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);

        tauGridCurrent[3][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);
    }

    gsl_matrix_complex_set_zero(tmp2);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);





    // tau[4] = Sum(mu = 1 to 4) Lambda(p_f).[gamma[mu], gamma[nu]].Lambda(p_i) (x) Lambda(k_f).[gamma[mu], gamma[nu]].Lambda(k_i)
    // Note: prefactor 1/8 here instead of symmetric/asymmetric basis
    tauGridCurrent[4][impulseIdx].setZero();
    for(int mu = 0; mu < 4; mu++)
    {
        for(int nu = 0; nu < 4; nu++)
        {
            gsl_matrix_complex_set_zero(tmp2);
            Commutator::commutator(DiracStructuresHelper::diracStructures.gamma[mu],
                                   DiracStructuresHelper::diracStructures.gamma[nu],
                                   tmp2);

            matProd3Elem(Lambda_pf, tmp2, Lambda_pi, tmp1, tmp_down);
            matProd3Elem(Lambda_kf, tmp2, Lambda_ki, tmp1, tmp_up);

            tauGridCurrent[4][impulseIdx] += Tensor4<4, 4, 4, 4>(tmp_down, tmp_up);
        }
    }

    tauGridCurrent[4][impulseIdx] = tauGridCurrent[4][impulseIdx] * gsl_complex_rect(0.125, 0);

    gsl_matrix_complex_set_zero(tmp_down);
    gsl_matrix_complex_set_zero(tmp_up);





    // Free temporary vars
    gsl_matrix_complex_free(tmp1);
    gsl_matrix_complex_free(tmp2);

    gsl_matrix_complex_free(tmp_down);
    gsl_matrix_complex_free(tmp_up);


    // Free projectors
    gsl_matrix_complex_free(Lambda_ki);
    gsl_matrix_complex_free(Lambda_kf);
    gsl_matrix_complex_free(Lambda_pi);
    gsl_matrix_complex_free(Lambda_pf);
}


void TensorBasis::calculateSymAsymBasis(int impulseIdx)
{
    // TODO check this works (operator overloading for Tensor might need adaption --> does the copy constructor work for return values?)

    // S1
    TGrid[0][impulseIdx] = tauGrid[0][impulseIdx] + tauGrid[1][impulseIdx] - tauGrid[4][impulseIdx] * gsl_complex_rect(1.0/3.0, 0);

    // S2
    TGrid[1][impulseIdx] = tauGrid[0][impulseIdx] - tauGrid[1][impulseIdx] + (tauGrid[2][impulseIdx] - tauGrid[3][impulseIdx]) * gsl_complex_rect(0.5, 0);

    // A1
    TGrid[2][impulseIdx] = tauGrid[0][impulseIdx] + tauGrid[1][impulseIdx] + tauGrid[4][impulseIdx];

    // A2
    TGrid[3][impulseIdx] = tauGrid[0][impulseIdx] - tauGrid[1][impulseIdx] - (tauGrid[2][impulseIdx] - tauGrid[3][impulseIdx]) * gsl_complex_rect(0.5, 0);

    // A3
    TGrid[4][impulseIdx] = tauGrid[2][impulseIdx] + tauGrid[3][impulseIdx];
}


void TensorBasis::matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C, gsl_matrix_complex* tmp, gsl_matrix_complex* res)
{
    gsl_matrix_complex_set_zero(tmp);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), A, B,
                   gsl_complex_rect(0, 0), tmp);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), tmp, C,
                   gsl_complex_rect(0, 0), res);
}


TensorBasis::TensorBasis(ExternalImpulseGrid* externalImpulseGrid, gsl_complex nucleon_mass) : len(externalImpulseGrid->getLength())
{
    tauGrid = new Tensor4<4, 4, 4, 4>* [basis_size];
    TGrid = new Tensor4<4, 4, 4, 4>* [basis_size];

    for(int i = 0; i < basis_size; i++)
    {
        tauGrid[i] = new Tensor4<4, 4, 4, 4>[len];
        TGrid[i] = new Tensor4<4, 4, 4, 4>[len];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        calculateBasis(impulseIdx, tauGrid, externalImpulseGrid->get_p_f(impulseIdx), externalImpulseGrid->get_p_i(impulseIdx),
                       externalImpulseGrid->get_k_f(impulseIdx),
                       externalImpulseGrid->get_k_i(impulseIdx));

        calculateSymAsymBasis(impulseIdx);
    }




    KMatrixGrid = new gsl_matrix_complex*[len];
    KInverseMatrixGrid = new gsl_matrix_complex*[len];

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        KMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(basis_size, basis_size);
        KInverseMatrixGrid[impulseIdx] = gsl_matrix_complex_alloc(basis_size, basis_size);

        calculateKMatrix(impulseIdx, basisGrid());
        calculateKMatrixInverse(impulseIdx);
    }
}

TensorBasis::~TensorBasis()
{
    for(int i = 0; i < basis_size; i++)
    {
        delete []tauGrid[i];
        delete []TGrid[i];
    }

    for(int impulseIdx = 0; impulseIdx < len; impulseIdx++)
    {
        gsl_matrix_complex_free(KMatrixGrid[impulseIdx]);
        gsl_matrix_complex_free(KInverseMatrixGrid[impulseIdx]);
    }

    delete []tauGrid;
    delete []TGrid;
    delete []KMatrixGrid;
    delete []KInverseMatrixGrid;
}

int TensorBasis::getTensorBasisElementCount() const
{
    return basis_size;
}

void TensorBasis::calculateKMatrix(int impulseIdx, Tensor4<4, 4, 4, 4>** basis)
{
    for(int i = 0; i < getTensorBasisElementCount(); i++)
    {
        for(int j = 0; j < getTensorBasisElementCount(); j++)
        {
            // TODO check leftContractWith is the correct thing to use
            gsl_matrix_complex_set(KMatrixGrid[impulseIdx], j, i,
                                   basis[i][impulseIdx].leftContractWith(&basis[j][impulseIdx]));
        }
    }

    /*
    // Set 0 what should be 0 (< 1E-10)
    double eps = 1E-10;
    for(int i = 0; i < KMatrixGrid[impulseIdx]->size1; i++)
    {
        for (int j = 0; j < KMatrixGrid[impulseIdx]->size2; j++)
        {
            gsl_complex entry = gsl_matrix_complex_get(KMatrixGrid[impulseIdx], i, j);

            if(!(abs(GSL_REAL(entry)) > eps || abs(GSL_IMAG(entry)) > eps))
                gsl_matrix_complex_set(KMatrixGrid[impulseIdx], i, j, GSL_COMPLEX_ZERO);
        }
    }
     */

    //std::cout << "K-Matrix: " << std::endl << PrintGSLElements::print_gsl_matrix_structure(KMatrixGrid[impulseIdx], 1E-10) << std::endl;
    //std::cout << "K-Matrix: " << std::endl << PrintGSLElements::print_gsl_matrix_complex(KMatrixGrid[impulseIdx]) << std::endl << std::endl;
}

void TensorBasis::calculateKMatrixInverse(int impulseIdx)
{
    int signum;
    gsl_permutation* p = gsl_permutation_alloc(basis_size);

    gsl_matrix_complex* LUDecomp = gsl_matrix_complex_alloc(basis_size, basis_size);
    gsl_matrix_complex_memcpy(LUDecomp, KMatrixGrid[impulseIdx]);

    gsl_linalg_complex_LU_decomp(LUDecomp, p, &signum);
    gsl_complex det = gsl_linalg_complex_LU_det(LUDecomp, signum);


    if(gsl_isnan(GSL_REAL(det)) || gsl_isnan(GSL_IMAG(det)) || gsl_complex_abs(det) == 0)
    {
        std::cout << "Singular Basis inversion matrix at impulseIdx " << impulseIdx << std::endl;
    }
    else
    {
        gsl_linalg_complex_LU_invert(LUDecomp, p, KInverseMatrixGrid[impulseIdx]);
    }


    // Set 0 what should be 0 (< 1E-10)
    /*
    double eps = 1E-15;
    for(int i = 0; i < KInverseMatrixGrid[impulseIdx]->size1; i++)
    {
        for (int j = 0; j < KInverseMatrixGrid[impulseIdx]->size2; j++)
        {
            gsl_complex entry = gsl_matrix_complex_get(KInverseMatrixGrid[impulseIdx], i, j);

            if(!(abs(GSL_REAL(entry)) > eps || abs(GSL_IMAG(entry)) > eps))
                gsl_matrix_complex_set(KInverseMatrixGrid[impulseIdx], i, j, GSL_COMPLEX_ZERO);
        }
    }
    */


    //std::cout << "K-Matrix Inverse: " << std::endl << PrintGSLElements::print_gsl_matrix_structure(KInverseMatrixGrid[impulseIdx], 1E-10) << std::endl;
    std::cout << "K-Matrix Inverse: " << std::endl << PrintGSLElements::print_gsl_matrix_complex(KInverseMatrixGrid[impulseIdx]) << std::endl;
}

gsl_matrix_complex* TensorBasis::K(int impulseIdx)
{
    return KMatrixGrid[impulseIdx];
}

gsl_matrix_complex* TensorBasis::KInv(int impulseIdx)
{
    return KInverseMatrixGrid[impulseIdx];
}

Tensor4<4, 4, 4, 4>** TensorBasis::basisGrid()
{
    if(BASIS == Basis::tau)
        return tauGrid;
    else if (BASIS == Basis::T)
        return TGrid;

    return nullptr;
}

Tensor4<4, 4, 4, 4>* TensorBasis::tau(int basisElemIdx, int externalImpulseIdx)
{
    return &(tauGrid[basisElemIdx][externalImpulseIdx]);
}

Tensor4<4, 4, 4, 4>* TensorBasis::T(int basisElemIdx, int externalImpulseIdx)
{
    return &(TGrid[basisElemIdx][externalImpulseIdx]);
}

Tensor4<4, 4, 4, 4>* TensorBasis::basisTensor(int basisElemIdx, int externalImpulseIdx)
{
    return &(basisGrid()[basisElemIdx][externalImpulseIdx]);
}




