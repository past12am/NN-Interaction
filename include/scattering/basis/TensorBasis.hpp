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

        Tensor4<4, 4, 4, 4>** tauGrid;
        Tensor4<4, 4, 4, 4>** tauGridTimelike;

        gsl_matrix_complex** KMatrixGrid;
        gsl_matrix_complex** KInverseMatrixGrid;

        void calculateBasis(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent, gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f, gsl_vector_complex* k_i, gsl_vector_complex* P);
        void matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C,  gsl_matrix_complex* tmp, gsl_matrix_complex* res);

        void calculateKMatrix(int impulseIdx, Tensor4<4, 4, 4, 4>** tauGridCurrent);
        void calculateKMatrixInverse(int impulseIdx);

    public:
        TensorBasis(ExternalImpulseGrid* externalImpulseGrid);
        virtual ~TensorBasis();

        int getTensorBasisElementCount() const;
        Tensor4<4, 4, 4, 4>* tauGridAt(int basisElemIdx);


        explicit operator std::string() const;

        Tensor4<4, 4, 4, 4>* tau(int basisElemIdx, int externalImpulseIdx);
        gsl_matrix_complex* K(int impulseIdx);
        gsl_matrix_complex* KInv(int impulseIdx);
};


#endif //NNINTERACTION_TENSORBASIS_HPP
