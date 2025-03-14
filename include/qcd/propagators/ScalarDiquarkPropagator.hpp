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
        static double D0;   // D(0)
        static double Doo;  // D(oo)
        static double L2;   // Second pole at x = -L2
        static double D1;   // Fit parameter

        static double m_sc;

        gsl_complex calc_p2_with_epsilon_shift(gsl_complex x_4, double A_imag, double D_plusminus, double epsilon);

        static double A_imag(double X, double eta);
        static double D_plusminus(double x2, double absx, double y, double phi, double X, double Z, bool sign_plus);
        static double Omega(double y, double phi, double Z);

    public:

        void D(gsl_vector_complex* p, gsl_complex* diquarkPropScalar);
        void D(gsl_complex x_4, double absx, double y, double phi, bool sign_plus, double X, double Z, double eta, gsl_complex* diquarkPropScalar, double epsilon);
};


#endif //NNINTERACTION_SCALARDIQUARKPROPAGATOR_HPP
