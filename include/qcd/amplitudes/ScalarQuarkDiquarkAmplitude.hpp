//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_SCALARQUARKDIQUARKAMPLITUDE_HPP
#define NNINTERACTION_SCALARQUARKDIQUARKAMPLITUDE_HPP


#include "../../operators/Projectors.hpp"

#include <complex>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex_math.h>

class ScalarQuarkDiquarkAmplitude
{
    private:
        gsl_complex c1 = gsl_complex_rect(0.7, 0);
        gsl_complex c2 = gsl_complex_rect(-0.2, 0);
        gsl_complex c3 = gsl_complex_rect(1.6, 0);

        gsl_vector_complex* p_copy;
        gsl_vector_complex* P_copy;

        gsl_matrix_complex* posEnergyProj;

        gsl_complex f(gsl_complex p2);

    public:
        ScalarQuarkDiquarkAmplitude();
        virtual ~ScalarQuarkDiquarkAmplitude();

        /*!
         * Calculates the matrix representing the Scalar Quark Diquark amplitude for given p and P
         *
         * @param p relative momentum between quark and diquark
         * @param P total nucleon momentum on the mass shell (P^2 = -M^2)
         * @param quarkDiquarkAmp return value for the generated propagator matrix
         */
        void Gamma(gsl_vector_complex* p, gsl_vector_complex* P, bool chargeConj, int threadIdx, gsl_matrix_complex* quarkDiquarkAmp);
};


#endif //NNINTERACTION_SCALARQUARKDIQUARKAMPLITUDE_HPP
