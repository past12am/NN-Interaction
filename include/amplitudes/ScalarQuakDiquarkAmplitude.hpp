//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_SCALARQUAKDIQUARKAMPLITUDE_HPP
#define NNINTERACTION_SCALARQUAKDIQUARKAMPLITUDE_HPP


#include <complex>
#include <gsl/gsl_matrix.h>
#include "../utils/ProjectorsHelper.hpp"

class ScalarQuakDiquarkAmplitude
{
    private:
        gsl_complex c1 = gsl_complex_rect(0.7, 0);
        gsl_complex c2 = gsl_complex_rect(-0.2, 0);
        gsl_complex c3 = gsl_complex_rect(1.6, 0);

        gsl_matrix_complex* posEnergyProj;

        ProjectorsHelper projectors;

        void Phi(gsl_vector_complex* p, gsl_vector_complex* P, gsl_matrix_complex* quarkDiquarkAmp);
        gsl_complex f(gsl_complex p2);

    public:
        ScalarQuakDiquarkAmplitude();

        virtual ~ScalarQuakDiquarkAmplitude();
};


#endif //NNINTERACTION_SCALARQUAKDIQUARKAMPLITUDE_HPP
