//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_TENSORBASIS_HPP
#define NNINTERACTION_TENSORBASIS_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

#include "../../utils/tensor/Tensor4.hpp"
#include "../../operators/Projectors.hpp"
#include "../impulse/ExternalImpulseGrid.hpp"


class TensorBasis
{
    private:
        int len;
        static const int basis_size = 5;

        Tensor4<4, 4, 4, 4>** tauGrid;
        Tensor4<4, 4, 4, 4>** tauPrimeGrid;
        Tensor4<4, 4, 4, 4>** TGrid;

        gsl_matrix_complex** KMatrixGrid;
        gsl_matrix_complex** KInverseMatrixGrid;
        gsl_matrix_complex** RInverseMatrixGrid;

        void calculateBasis(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent, gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f, gsl_vector_complex* k_i, bool build_charge_conj_tensors);
        void calculateSymAsymBasis(int impulseIdx);

        void matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C,  gsl_matrix_complex* tmp, gsl_matrix_complex* res);

        void calculateKMatrix(int impulseIdx, Tensor4<4, 4, 4, 4>** basis);
        void calculateKMatrixInverse(int impulseIdx);

        void calculateRMatrixInverse(gsl_matrix_complex* RInvMatrixCur, double X, double Z);

        Tensor4<4, 4, 4, 4>* tau(int basisElemIdx, int externalImpulseIdx);
        Tensor4<4, 4, 4, 4>* tauPrime(int basisElemIdx, int externalImpulseIdx);
        Tensor4<4, 4, 4, 4>* T(int basisElemIdx, int externalImpulseIdx);

        Tensor4<4, 4, 4, 4>** basisGrid();
        Tensor4<4, 4, 4, 4>** basisProjectionGrid();

        static double calc_c(double X, double Z);
        static double calc_d(double X, double Z);

        static double calc_n_plus(double c, double d);
        static double calc_n_minus(double c, double d);
        static double calc_g(double c, double n_minus);
        static double calc_h(double n_minus, double c);
        static double calc_k(double g, double n_minus);

        static double calc_e1(double c, double d, double n_plus, double n_minus);
        static double calc_e2(double c, double d, double k, double n_plus);
        static double calc_e3(double c, double d, double n_plus, double n_minus);
        static double calc_e4(double c, double d, double n_plus);
        static double calc_e5(double c, double d, double n_plus);

    public:
        TensorBasis(ExternalImpulseGrid* externalImpulseGrid, gsl_complex nucleon_mass);
        virtual ~TensorBasis();

        int getTensorBasisElementCount() const;

        Tensor4<4, 4, 4, 4>* basisTensor(int basisElemIdx, int externalImpulseIdx);
        Tensor4<4, 4, 4, 4>* basisTensorProjection(int basisElemIdx, int externalImpulseIdx);

        gsl_matrix_complex* K(int impulseIdx);
        gsl_matrix_complex* KInv(int impulseIdx);

        gsl_matrix_complex* RInv(int impulseIdx);
};


#endif //NNINTERACTION_TENSORBASIS_HPP
