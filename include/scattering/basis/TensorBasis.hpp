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

        ExternalImpulseGrid* externalImpulseGrid;

        void calculateBasis(int impulseIdx, gsl_vector_complex* p_f, gsl_vector_complex* p_i, gsl_vector_complex* k_f, gsl_vector_complex* k_i, gsl_vector_complex* P, gsl_vector_complex* K);
        void matProd3Elem(const gsl_matrix_complex* A, const gsl_matrix_complex* B, const gsl_matrix_complex* C,  gsl_matrix_complex* tmp, gsl_matrix_complex* res);

    public:
        TensorBasis(ExternalImpulseGrid* externalImpulseGrid);
        virtual ~TensorBasis();

        int getLength() const;
        Tensor4<4, 4, 4, 4>* tauGridAt(int basisElemIdx);


        explicit operator std::string() const;


        Tensor4<4, 4, 4, 4>* tau(int basisElemIdx, int externalImpulseIdx);
};


#endif //NNINTERACTION_TENSORBASIS_HPP
