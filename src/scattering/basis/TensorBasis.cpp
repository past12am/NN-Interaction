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
    /*
     * Note the differnce for the Lambda_pf via Mathematica and C++ (they should be equal)
     * C++ PrintGSLElements::print_gsl_matrix_complex(Lambda_pf)    =
     *      [     1.0076 +i 0          ,           0 +i 0          ,           0 -i 0.15304    ,           0 +i 0          ]
     *      [          0 +i 0          ,      1.0076 +i 0          ,           0 +i 0          ,           0 +i 0.15304    ]
     *      [          0 +i 0.15304    ,           0 +i 0          ,  -0.0075896 +i 0          ,           0 +i 0          ]
     *      [          0 +i 0          ,           0 -i 0.15304    ,           0 +i 0          ,  -0.0075896 +i 0          ]
     *
     * Mathematica:
     *      {{0.978714,         0,                  0. - 0.144338 I,     0},
     *       {0,                0.978714,           0,                   0. + 0.144338 I},
     *       {0. + 0.144338 I,  0,                  0.0212864,           0},
     *       {0,                0. - 0.144338 I,    0,                   0.0212864}}
     *
     *
     *
     * However, the p_f from C++ and Mathematica match
     *         Mathematica      C++
     *          0                   0 +i 0
     *          0                   0 +i 0
     *          0.297254            0.297254 +i 0
     *          0.98588             0.98588 +i 0
     */

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

TensorBasis::TensorBasis(ExternalImpulseGrid* externalImpulseGrid, gsl_complex nucleon_mass) : len(externalImpulseGrid->getLength())
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


        //calculateBasis(impulseIdx, tauGridTimelike, externalImpulseGrid->get_p_f_timelike(impulseIdx), externalImpulseGrid->get_p_i_timelike(impulseIdx),
        //               externalImpulseGrid->get_k_f_timelike(impulseIdx),
        //               externalImpulseGrid->get_k_i_timelike(impulseIdx), externalImpulseGrid->get_P_timelike(impulseIdx));

        //calculateKMatrix(impulseIdx, tauGrid);
        //calculateKMatrixInverse(impulseIdx);
    }

    for(int X_idx = 0; X_idx < externalImpulseGrid->getLenX(); X_idx++)
    {
        for(int z_idx = 0; z_idx < externalImpulseGrid->getLenZ(); z_idx++)
        {
            double X = externalImpulseGrid->getXAt(X_idx);
            double z = externalImpulseGrid->getZAt(z_idx);
            double M = GSL_REAL(nucleon_mass);

            // @Commit: investigate: tauGrid[0].tensor[0][0][0][0] is different from mathematica expression by ~ 0.1                is fixed now
            /*
             * \[Tau]1\[LetterSpace]0 = \[Tau][1] /. { a -> 1, M -> 0.93999999999999995, X -> 0.10000000000000001, z -> -0.99990000000000001} // Simplify ;
             * \[Tau]1\[LetterSpace]0[[1, 1, 1, 1]]
             *      = 0.878061
             *
             *      Note: same goes for z = 0, and therefore most probably for all grid points
             *
             *
             * tau(0, externalImpulseGrid->getGridIdx(X_idx, z_idx)).tensor[0][0][0][0] = 0.9837001396596482
             */
            int impulseIdx = externalImpulseGrid->getGridIdx(X_idx, z_idx);
            calculateKMatrixInverseAnalytic(KInverseMatrixGrid[impulseIdx], X, z, M, externalImpulseGrid->get_a());
        }
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

void TensorBasis::calculateKMatrixInverseAnalytic(gsl_matrix_complex* KInv, double X, double z, double M, double a)
{
    // TODO check used pow is not too inaccurate vs gsl_pow

    // [0, x]
    double entry_00 = (4.0 * gsl_pow_4(a + X + a * X) * (4.0 * gsl_pow_3(a) * gsl_pow_3(1.0 + X)
                        - 4.0 * gsl_pow_2(a) * X * gsl_pow_2(1.0 + X) * z
                        + gsl_pow_3(X) * gsl_pow_2(1.0 + z)
                        + a * gsl_pow_2(X) * (1.0 + X) * (-7.0 - 2.0 * z + 5.0 * gsl_pow_2(z)))) /
                      (a * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z)));


    double entry_01 = -((4.0 * pow(a + X + a * X, 4.5) * (2.0 * a * (1.0 + X) - X * (1.0 + z))) /
                       (a * M * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1 + z) * gsl_pow_3(2 * a * (1.0 + X) + X * (1.0 + z))));

    // [1, x]
    double entry_11 = (2.0 * gsl_pow_4(a + X + a * X) * (gsl_pow_3(X) * (-1.0 + z) * gsl_pow_2(1.0 + z)
                                                            + 4 * gsl_pow_3(a) * gsl_pow_3(1.0 + X) * (3.0 + 5.0 * z)
                                                            + 4 * gsl_pow_2(a) * X * gsl_pow_2(1.0 + X) * (5.0 + 9.0 * z + 2.0 * gsl_pow_2(z))
                                                            + a * gsl_pow_2(X) * (1.0 + X) * (9.0 + 13.0 * z + 7.0 * gsl_pow_2(z) + 3.0 * gsl_pow_3(z)))) /
                      (gsl_pow_2(a) * gsl_pow_2(M) * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * (-1.0 + z) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_12 = -((pow(a + X + a * X, 4.5) * (8.0 * a * X * (1.0 + X) * (1.0 + z)
                                                    + gsl_pow_2(X) * (-1.0 + z) * gsl_pow_2(1.0 + z)
                                                    + 4.0 * gsl_pow_2(a) * gsl_pow_2(1.0 + X) * (1.0 + 3.0 * z))) /
                       (gsl_pow_2(a) * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * (-1.0 + z) * gsl_pow_2(1.0 + z) * gsl_pow_3(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    double entry_14 = -((16.0 * pow(a + X + a * X, 5.5) * (2.0 * a * (1.0 + X) - X * (1.0 + z))) /
                        (a * M * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    double entry_16 = (pow(a + X + a * X, 4.5) * (2.0 * a * (1.0 + X) - X * (1.0 + z))) /
                      (gsl_pow_2(a) * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_17 = -((gsl_pow_5(a + X + a * X) * gsl_pow_2(-2.0 * a * (1.0 + X) + X * (1.0 + z))) /
                        (gsl_pow_2(a * M * X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    // [2, x]
    double entry_22 = (gsl_pow_4(a + X + a * X) * (8.0 * a * X * (1.0 + X) * (1.0 + z)
                                                      + gsl_pow_2(X) * (-1.0 + z) * gsl_pow_2(1 + z)
                                                      + 4.0 * gsl_pow_2(a) * gsl_pow_2(1.0 + X) * (1.0 + 3.0 * z))) /
                      (2.0 * gsl_pow_2(a * X) * gsl_pow_2(1.0 + X) * (-1.0 + z) * gsl_pow_2(1.0 + z) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_24 = (8.0 * gsl_pow_5(a + X + a * X) * (2.0 * a * (1.0 + X) - X * (1 + z))) /
                      (a * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1.0 + z) * gsl_pow_3(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_26 = -((gsl_pow_4(a + X + a * X) * (2.0 * a * (1.0 + X) - X * (1.0 + z))) /
                        (2.0 * gsl_pow_2(a * X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1 + z) * (2.0 * a * (1.0 + X) + X * (1.0 + z))));

    double entry_27 = (pow(a + X + a * X, 4.5) * gsl_pow_2(-2.0 * a * (1.0 + X) + X * (1.0 + z))) /
                      (2.0 * gsl_pow_2(a) * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_3(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    // [3, x]
    double entry_33 = -(gsl_pow_4(a + X + a * X) /
                       (a * gsl_pow_2(M * X) * (1.0 + X) * (-1.0 + gsl_pow_2(z)) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    // [4, x]
    double entry_44 = (4.0 * gsl_pow_4(a + X + a * X) * (gsl_pow_3(X) * (-1.0 + z) * gsl_pow_2(1.0 + z)
                                                         + 4.0 * gsl_pow_3(a) * gsl_pow_3(1.0 + X) * (7.0 + 9.0 * z)
                                                         + 4.0 * gsl_pow_2(a) * X * gsl_pow_2(1 + X) * (12.0 + 17.0 * z + 3.0 * gsl_pow_2(z))
                                                         + a * gsl_pow_2(X) * (1.0 + X) * (23.0 + 27.0 * z + 9.0 * gsl_pow_2(z)
                                                         + 5.0 * gsl_pow_3(z)))) /
                      (a * gsl_pow_2(X) * (1.0 + X) * gsl_pow_2(-1.0 + z) * (1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_46 = -((4.0 * gsl_pow_5(a + X + a * X)) /
                        (a * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1.0 + z) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    double entry_47 = (2.0 * pow(a + X + a * X, 4.5) * (4.0 * gsl_pow_2(a) * gsl_pow_2(1.0 + X)
                                                        - 4.0 * a * X * (1.0 + X) * (-1.0 + z)
                                                        + gsl_pow_2(X) * (-3.0 - 2.0 * z + gsl_pow_2(z)))) /
                      (a * M * gsl_pow_2(X) * (1.0 + X) * (-1.0 + z) * (1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    // [5, x]
    double entry_55 = -((4.0 * gsl_pow_4(a + X + a * X)) /
                        (a * gsl_pow_2(M) * gsl_pow_2(X) * (1.0 + X) * (-1.0 + gsl_pow_2(z)) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z))));

    // [6, x]
    double entry_66 = (gsl_pow_4(a + X + a * X) * (4.0 * gsl_pow_2(a) * gsl_pow_2(1.0 + X) + gsl_pow_2(X) * gsl_pow_2(1.0 + z))) /
                      (2.0 * gsl_pow_2(a * X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    double entry_67 = (pow(a + X + a * X, 4.5) * (-2.0 * a * (1.0 + X) + X * (1.0 + z))) /
                      (2.0 * gsl_pow_2(a) * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_2(2.0 * a * (1.0 + X) + X * (1.0 + z)));

    // [7, x]
    double entry_77 = (gsl_pow_4(a + X + a * X) * (4.0 * gsl_pow_2(a) * X * gsl_pow_2(1.0 + X)
                                                      + 4.0 * gsl_pow_3(a) * gsl_pow_3(1.0 + X)
                                                      + gsl_pow_3(X) * gsl_pow_2(1.0 + z)
                                                      + a * gsl_pow_2(X) * (1.0 + X) * (-1.0 + 2.0 * z + 3.0 * gsl_pow_2(z)))) /
                      (2.0 * gsl_pow_2(a * M * X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 * a * (1.0 + X) + X * (1.0 + z)));


    // Set matrix entries
    gsl_matrix_complex_set_zero(KInv);

    gsl_matrix_complex_set(KInv, 0, 0, gsl_complex_rect(entry_00, 0));
    gsl_matrix_complex_set(KInv, 1, 1, gsl_complex_rect(entry_11, 0));
    gsl_matrix_complex_set(KInv, 2, 2, gsl_complex_rect(entry_22, 0));
    gsl_matrix_complex_set(KInv, 3, 3, gsl_complex_rect(entry_33, 0));
    gsl_matrix_complex_set(KInv, 4, 4, gsl_complex_rect(entry_44, 0));
    gsl_matrix_complex_set(KInv, 5, 5, gsl_complex_rect(entry_55, 0));
    gsl_matrix_complex_set(KInv, 6, 6, gsl_complex_rect(entry_66, 0));
    gsl_matrix_complex_set(KInv, 7, 7, gsl_complex_rect(entry_77, 0));


    gsl_matrix_complex_set(KInv, 0, 1, gsl_complex_rect(entry_01, 0));
    gsl_matrix_complex_set(KInv, 1, 0, gsl_complex_rect(entry_01, 0));


    gsl_matrix_complex_set(KInv, 1, 2, gsl_complex_rect(entry_12, 0));
    gsl_matrix_complex_set(KInv, 2, 1, gsl_complex_rect(entry_12, 0));

    gsl_matrix_complex_set(KInv, 1, 4, gsl_complex_rect(entry_14, 0));
    gsl_matrix_complex_set(KInv, 4, 1, gsl_complex_rect(entry_14, 0));

    gsl_matrix_complex_set(KInv, 1, 6, gsl_complex_rect(entry_16, 0));
    gsl_matrix_complex_set(KInv, 6, 1, gsl_complex_rect(entry_16, 0));

    gsl_matrix_complex_set(KInv, 1, 7, gsl_complex_rect(entry_17, 0));
    gsl_matrix_complex_set(KInv, 7, 1, gsl_complex_rect(entry_17, 0));


    gsl_matrix_complex_set(KInv, 2, 4, gsl_complex_rect(entry_24, 0));
    gsl_matrix_complex_set(KInv, 4, 2, gsl_complex_rect(entry_24, 0));

    gsl_matrix_complex_set(KInv, 2, 6, gsl_complex_rect(entry_26, 0));
    gsl_matrix_complex_set(KInv, 6, 2, gsl_complex_rect(entry_26, 0));

    gsl_matrix_complex_set(KInv, 2, 7, gsl_complex_rect(entry_27, 0));
    gsl_matrix_complex_set(KInv, 7, 2, gsl_complex_rect(entry_27, 0));


    gsl_matrix_complex_set(KInv, 4, 6, gsl_complex_rect(entry_46, 0));
    gsl_matrix_complex_set(KInv, 6, 4, gsl_complex_rect(entry_46, 0));

    gsl_matrix_complex_set(KInv, 4, 7, gsl_complex_rect(entry_47, 0));
    gsl_matrix_complex_set(KInv, 7, 4, gsl_complex_rect(entry_47, 0));


    gsl_matrix_complex_set(KInv, 6, 7, gsl_complex_rect(entry_67, 0));
    gsl_matrix_complex_set(KInv, 7, 6, gsl_complex_rect(entry_67, 0));
}

void TensorBasis::calculateKMatrixInverseAnalyticTimelikeq(gsl_matrix_complex* KInv, double X, double z, double M)
{
    // Note that the used analytic expressions are for timelike q
    //          --> see calculateKMatrixInverseAnalytic

    //[0,0]
    double entry_00_real = (4.0 * (4.0 + 4.0 * X * (3 + z) + 4 * gsl_pow_3(X) * (-1 + gsl_pow_2(z)) + gsl_pow_2(X) * (5.0 + 6.0 * z + 5.0 * gsl_pow_2(z)))) /
                (gsl_pow_2(X) * (1.0 + X) * gsl_pow_4(2.0 + X - X * z) * (-1.0 + gsl_pow_2(z)));

    //[0,1]
    double entry_01_imag = -((4.0 * (2.0 + X * (3.0 + z))) /
                            (M * gsl_pow_2(X) * (1.0 + X) * gsl_pow_3(-2.0 + X * (-1 + z)) * (-1 + gsl_pow_2(z))));


    //[1, 1]
    double entry_11_real = -((2.0 * (2.0 * gsl_pow_3(X) * gsl_pow_2(-1.0 + z) * (1.0 + z) + 4 * (3.0 + 5.0 * z) -
                                8.0 * X * (-2.0 - 3.0 * z + gsl_pow_2(z)) +
                                gsl_pow_2(X) * (5.0 + z - 9.0 * gsl_pow_2(z) + 3.0 * gsl_pow_3(z)))) /
                            (gsl_pow_2(M) * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * (-1.0 + z) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 + X - X * z)));

    //[1, 2]
    double entry_12_imag = -(((4.0 + 12.0 * z + 16.0 * X * z + gsl_pow_2(X) * (-5.0 + 3.0 * z + gsl_pow_2(z) + gsl_pow_3(z)))) /
                            (M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_3(-2.0 + X * (-1.0 + z)) * (-1.0 + z) * gsl_pow_2(1.0 + z)));


    //[2, 2]
    double entry_22_real = (4.0 + 12.0 * z + 16.0 * X * z + gsl_pow_2(X) * (-5.0 + 3.0 * z + gsl_pow_2(z) + gsl_pow_3(z))) /
                           (2.0 * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * (-1.0 + z) * gsl_pow_2(1.0 + z));


    //[3, 3]
    double entry_33_real = 1.0 / (gsl_pow_2(M) * gsl_pow_2(X) * (1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * (-1.0 + gsl_pow_2(z)));


    //[1, 4]
    double entry_14_imag = (16.0 * (2.0 + X * (3.0 + z))) / (M * gsl_pow_2(X) * (1.0 + X) * gsl_pow_4(2.0 + X - X * z) * (-1.0 + gsl_pow_2(z)));

    //[2, 4]
    double entry_24_real = -((8.0 * (2.0 + X * (3.0 + z))) / (gsl_pow_2(X) * (1.0 + X) * gsl_pow_3(-2.0 + X * (-1.0 + z)) * (-1.0 + gsl_pow_2(z))));

    //[4, 4]
    double entry_44_real = (4.0 * (4.0 * gsl_pow_3(X) * gsl_pow_2(-1.0 + z) * (1.0 + z) + 4.0 * (7.0 + 9.0 * z)
                                   + X * (36.0 + 40.0 * z - 12.0 * gsl_pow_2(z))
                                   + gsl_pow_2(X) * (11.0 - z - 15.0 * gsl_pow_2(z) + 5.0 * gsl_pow_3(z)))) /
                            (gsl_pow_2(X) * (1.0 + X) * gsl_pow_2(-1.0 + z) * (1.0 + z) * gsl_pow_4(2.0 + X - X * z));


    //[5, 5]
    double entry_55_real = 4.0 / (gsl_pow_2(M) * gsl_pow_2(X) * (1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * (-1.0 +gsl_pow_2(z)));


    //[6, 6]
    double entry_66_real = (4.0 + 8.0 * X + gsl_pow_2(X) * (5.0 + 2.0 * z + gsl_pow_2(z))) /
                           (2.0 * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * gsl_pow_2(1.0 + z));

    //[1, 6]
    double entry_16_imag = -(((2.0 + X * (3.0 + z))) / (M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * gsl_pow_2(1.0 + z)));

    //[2, 6]
    double entry_26_real = (2.0 + X * (3.0 + z)) / (2.0 * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * (-2.0 + X * (-1.0 + z)) * gsl_pow_2(1.0 + z));

    //[4, 6]
    double entry_46_real = -(4.0 / (gsl_pow_2(X) * (1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * (-1.0 + gsl_pow_2(z))));


    //[7, 7]
    double entry_77_real = -((4.0 + 8.0 * X + 2.0 * gsl_pow_3(X) * (-1.0 + gsl_pow_2(z)) + gsl_pow_2(X) * (3.0 + 2.0 * z + 3.0 * gsl_pow_2(z))) /
                            (2.0 * gsl_pow_2(M) * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 + X - X * z)));

    //[1, 7]
    double entry_17_real = gsl_pow_2(2.0 + X * (3.0 + z)) / (gsl_pow_2(M) * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(1.0 + z) * gsl_pow_4(2.0 + X - X * z));

    //[2, 7]
    double entry_27_imag = (gsl_pow_2(2.0 + X * (3.0 + z))) / (2.0 * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_3(-2.0 + X * (-1.0 + z)) * gsl_pow_2(1.0 + z));

    //[4, 7]
    double entry_47_imag = -((2.0 * (4.0 + 4.0 * X * (1 + z) + gsl_pow_2(X) * (-3.0 + 2.0 * z + gsl_pow_2(z)))) /
                            (M * gsl_pow_2(X) * (1.0 + X) * gsl_pow_4(2.0 + X - X * z) * (-1.0 + gsl_pow_2(z))));

    //[6, 7]
    double entry_67_imag = ((2.0 + X * (3.0 + z))) / (2.0 * M * gsl_pow_2(X) * gsl_pow_2(1.0 + X) * gsl_pow_2(-2.0 + X * (-1.0 + z)) * gsl_pow_2(1.0 + z));


    // Set matrix entries
    gsl_matrix_complex_set_zero(KInv);

    gsl_matrix_complex_set(KInv, 0, 0, gsl_complex_rect(entry_00_real, 0));
    gsl_matrix_complex_set(KInv, 1, 1, gsl_complex_rect(entry_11_real, 0));
    gsl_matrix_complex_set(KInv, 2, 2, gsl_complex_rect(entry_22_real, 0));
    gsl_matrix_complex_set(KInv, 3, 3, gsl_complex_rect(entry_33_real, 0));
    gsl_matrix_complex_set(KInv, 4, 4, gsl_complex_rect(entry_44_real, 0));
    gsl_matrix_complex_set(KInv, 5, 5, gsl_complex_rect(entry_55_real, 0));
    gsl_matrix_complex_set(KInv, 6, 6, gsl_complex_rect(entry_66_real, 0));
    gsl_matrix_complex_set(KInv, 7, 7, gsl_complex_rect(entry_77_real, 0));


    gsl_matrix_complex_set(KInv, 0, 1, gsl_complex_rect(0, entry_01_imag));
    gsl_matrix_complex_set(KInv, 1, 0, gsl_complex_rect(0, entry_01_imag));


    gsl_matrix_complex_set(KInv, 1, 2, gsl_complex_rect(0, entry_12_imag));
    gsl_matrix_complex_set(KInv, 2, 1, gsl_complex_rect(0, entry_12_imag));

    gsl_matrix_complex_set(KInv, 1, 4, gsl_complex_rect(0, entry_14_imag));
    gsl_matrix_complex_set(KInv, 4, 1, gsl_complex_rect(0, entry_14_imag));

    gsl_matrix_complex_set(KInv, 1, 6, gsl_complex_rect(0, entry_16_imag));
    gsl_matrix_complex_set(KInv, 6, 1, gsl_complex_rect(0, entry_16_imag));

    gsl_matrix_complex_set(KInv, 1, 7, gsl_complex_rect(entry_17_real, 0));
    gsl_matrix_complex_set(KInv, 7, 1, gsl_complex_rect(entry_17_real, 0));


    gsl_matrix_complex_set(KInv, 2, 4, gsl_complex_rect(entry_24_real, 0));
    gsl_matrix_complex_set(KInv, 4, 2, gsl_complex_rect(entry_24_real, 0));

    gsl_matrix_complex_set(KInv, 2, 6, gsl_complex_rect(entry_26_real, 0));
    gsl_matrix_complex_set(KInv, 6, 2, gsl_complex_rect(entry_26_real, 0));

    gsl_matrix_complex_set(KInv, 2, 7, gsl_complex_rect(0, entry_27_imag));
    gsl_matrix_complex_set(KInv, 7, 2, gsl_complex_rect(0, entry_27_imag));


    gsl_matrix_complex_set(KInv, 4, 6, gsl_complex_rect(entry_46_real, 0));
    gsl_matrix_complex_set(KInv, 6, 4, gsl_complex_rect(entry_46_real, 0));

    gsl_matrix_complex_set(KInv, 4, 7, gsl_complex_rect(0, entry_47_imag));
    gsl_matrix_complex_set(KInv, 7, 4, gsl_complex_rect(0, entry_47_imag));


    gsl_matrix_complex_set(KInv, 6, 7, gsl_complex_rect(0, entry_67_imag));
    gsl_matrix_complex_set(KInv, 7, 6, gsl_complex_rect(0, entry_67_imag));
}

void TensorBasis::calculateKMatrixInverse(int impulseIdx)
{
    int signum;
    gsl_permutation* p = gsl_permutation_alloc(8);

    gsl_matrix_complex* LUDecomp = gsl_matrix_complex_alloc(8, 8);
    gsl_matrix_complex_memcpy(LUDecomp, KMatrixGrid[impulseIdx]);

    gsl_linalg_complex_LU_decomp(LUDecomp, p, &signum);
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




