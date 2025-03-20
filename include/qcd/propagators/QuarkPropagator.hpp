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
        gsl_complex sigma_v(gsl_complex p2, gsl_complex x_4, double absx, double y, double phi, double X, double Z, bool sign_plus, double eta, double epsilon);

        gsl_complex sigma_s(gsl_complex M, gsl_complex sigma_v);
        gsl_complex M(gsl_complex p2);

        gsl_complex Z_f(gsl_complex p2, gsl_complex M2);

        gsl_complex calc_p2_with_epsilon_shift(gsl_complex x_4, double A_imag, double D_plusminus, double epsilon);

        static double A_imag(double X, double eta);
        static double D_plusminus(double x2, double absx, double y, double phi, double X, double Z, bool sign_plus);
        static double Omega(double y, double phi, double Z);

    public:
        QuarkPropagator();
        virtual ~QuarkPropagator();

        void S(gsl_vector_complex* p, gsl_matrix_complex* quarkProp);
        void S(gsl_vector_complex* p, gsl_matrix_complex* quarkProp, gsl_complex x_4, double absx, double y, double phi, bool sign_plus, double X, double Z, double eta, double epsilon);

};


#endif //NNINTERACTION_QUARKPROPAGATOR_HPP
