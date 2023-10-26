//
// Created by past12am on 8/18/23.
//

#ifndef NNINTERACTION_EXTERNALIMPULSEGRID_HPP
#define NNINTERACTION_EXTERNALIMPULSEGRID_HPP

#include <complex>
#include <gsl/gsl_vector.h>
#include "ZXGrid.hpp"

class ExternalImpulseGrid : public ZXGrid
{
    private:
        gsl_complex nucleon_mass;

        gsl_vector_complex** q_ext;
        gsl_vector_complex** p_ext;
        gsl_vector_complex** k_ext;

        gsl_vector_complex** q_ext_timelike;

        gsl_vector_complex** Q;
        gsl_vector_complex** P;
        gsl_vector_complex** K;

        gsl_vector_complex** P_timelike;

        gsl_vector_complex** p_i;
        gsl_vector_complex** p_f;
        gsl_vector_complex** k_i;
        gsl_vector_complex** k_f;

        gsl_vector_complex** p_i_timelike;
        gsl_vector_complex** p_f_timelike;
        gsl_vector_complex** k_i_timelike;
        gsl_vector_complex** k_f_timelike;

        void calc_k_ext(gsl_vector_complex* k_ext, double X, gsl_complex nucleon_mass, double z);
        void calc_q_ext(gsl_vector_complex* q_ext, double X, gsl_complex nucleon_mass, double a);
        void calc_p_ext(gsl_vector_complex* p_ext, double X, gsl_complex nucleon_mass);

        void calc_q_ext_timelike(gsl_vector_complex* q_ext_timelike, double X, gsl_complex nucleon_mass);

        void calc_Q(gsl_vector_complex* Q, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);
        void calc_P(gsl_vector_complex* P, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);
        void calc_K(gsl_vector_complex* K, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);

        void calc_p_i(gsl_vector_complex* p_i, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);
        void calc_p_f(gsl_vector_complex* p_f, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);
        void calc_k_i(gsl_vector_complex* k_i, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);
        void calc_k_f(gsl_vector_complex* k_f, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext);

    public:
        ExternalImpulseGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, double a);
        virtual ~ExternalImpulseGrid();

        gsl_vector_complex* get_Q(int idx);
        gsl_vector_complex* get_P(int idx);
        gsl_vector_complex* get_K(int idx);

        gsl_vector_complex* get_P_timelike(int idx);

        gsl_vector_complex* get_p_i(int idx);
        gsl_vector_complex* get_p_f(int idx);
        gsl_vector_complex* get_k_i(int idx);
        gsl_vector_complex* get_k_f(int idx);

        gsl_vector_complex* get_p_i_timelike(int idx);
        gsl_vector_complex* get_p_f_timelike(int idx);
        gsl_vector_complex* get_k_i_timelike(int idx);
        gsl_vector_complex* get_k_f_timelike(int idx);

        double calc_tau(int XIdx, int zIdx);
};


#endif //NNINTERACTION_EXTERNALIMPULSEGRID_HPP
