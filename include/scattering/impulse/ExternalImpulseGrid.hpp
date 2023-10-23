//
// Created by past12am on 8/18/23.
//

#ifndef NNINTERACTION_EXTERNALIMPULSEGRID_HPP
#define NNINTERACTION_EXTERNALIMPULSEGRID_HPP

#include <complex>
#include <gsl/gsl_vector.h>
#include "ZTauGrid.hpp"

class ExternalImpulseGrid : public ZTauGrid
{
    private:
        gsl_vector_complex** Q;
        gsl_vector_complex** P;
        gsl_vector_complex** K;

        gsl_vector_complex** p_i;
        gsl_vector_complex** p_f;
        gsl_vector_complex** k_i;
        gsl_vector_complex** k_f;

        void calc_Q(gsl_vector_complex* Q, double tau, gsl_complex M_nucleon);
        void calc_P(gsl_vector_complex* P, double tau, gsl_complex M_nucleon, gsl_complex a);
        void calc_K(gsl_vector_complex* K, double tau, double z, gsl_complex M_nucleon, gsl_complex a);

        void calc_p_i(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_i);
        void calc_p_f(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_f);
        void calc_k_i(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_i);
        void calc_k_f(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_f);

    public:
        ExternalImpulseGrid(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex M_nucleon, gsl_complex a);
        virtual ~ExternalImpulseGrid();

        gsl_vector_complex* get_Q(int idx);
        gsl_vector_complex* get_P(int idx);
        gsl_vector_complex* get_K(int idx);

        gsl_vector_complex* get_p_i(int idx);
        gsl_vector_complex* get_p_f(int idx);
        gsl_vector_complex* get_k_i(int idx);
        gsl_vector_complex* get_k_f(int idx);
};


#endif //NNINTERACTION_EXTERNALIMPULSEGRID_HPP
