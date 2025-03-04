//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_QUARKPROPAGATOR_HPP
#define NNINTERACTION_QUARKPROPAGATOR_HPP

#include <gsl/gsl_matrix.h>


class QuarkPropagator
{
    private:
        gsl_matrix_complex* pSlashCurrent;

        gsl_complex sigma_v(gsl_complex p2);
        gsl_complex sigma_s(gsl_complex M, gsl_complex sigma_v);
        gsl_complex M(gsl_complex p2);

    public:
        QuarkPropagator();
        virtual ~QuarkPropagator();

        void S(gsl_vector_complex* p, gsl_matrix_complex* quarkProp);

};


#endif //NNINTERACTION_QUARKPROPAGATOR_HPP
