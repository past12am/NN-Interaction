//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_SCALARDIQUARKPROPAGATOR_HPP
#define NNINTERACTION_SCALARDIQUARKPROPAGATOR_HPP


#include <complex>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_matrix.h>

class ScalarDiquarkPropagator
{
    private:
        gsl_complex M(gsl_complex p2);

    public:

        void D(gsl_vector_complex* p, gsl_complex* diquarkPropScalar);

};


#endif //NNINTERACTION_SCALARDIQUARKPROPAGATOR_HPP
