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

        void calculateBasis(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent, gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f, gsl_vector_complex* k_i, bool build_charge_conj_tensors);
        void calculateSymAsymBasis(int impulseIdx);

        void matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C,  gsl_matrix_complex* tmp, gsl_matrix_complex* res);

        void calculateKMatrix(int impulseIdx, Tensor4<4, 4, 4, 4>** basis);
        void calculateKMatrixInverse(int impulseIdx);

        void calculateRMatrixInverse(int impulseIdx);

        Tensor4<4, 4, 4, 4>* tau(int basisElemIdx, int externalImpulseIdx);
        Tensor4<4, 4, 4, 4>* tauPrime(int basisElemIdx, int externalImpulseIdx);
        Tensor4<4, 4, 4, 4>* T(int basisElemIdx, int externalImpulseIdx);

    public:
        TensorBasis(ExternalImpulseGrid* externalImpulseGrid, gsl_complex nucleon_mass);
        virtual ~TensorBasis();

        int getTensorBasisElementCount() const;

        Tensor4<4, 4, 4, 4>* basisTensor(int basisElemIdx, int externalImpulseIdx);

        Tensor4<4, 4, 4, 4>** basisGrid();

        gsl_matrix_complex* K(int impulseIdx);
        gsl_matrix_complex* KInv(int impulseIdx);
};


#endif //NNINTERACTION_TENSORBASIS_HPP
