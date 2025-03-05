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
        // Set A
        //gsl_vector_complex** q_ext;
        //gsl_vector_complex** p_ext;
        //gsl_vector_complex** k_ext;


        // Set B
        gsl_vector_complex** l_ext;
        gsl_vector_complex** r_ext;
        gsl_vector_complex** P_ext;

        void calc_l_ext(gsl_vector_complex* l_ext, double X, double Z);
        void calc_r_ext(gsl_vector_complex* r_ext, double X, double Z);
        void calc_P_ext(gsl_vector_complex* P_ext, double X, double Z);


        gsl_vector_complex** p_i;
        gsl_vector_complex** p_f;
        gsl_vector_complex** k_i;
        gsl_vector_complex** k_f;

        void calc_p_i(gsl_vector_complex* p_i, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext);
        void calc_p_f(gsl_vector_complex* p_f, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext);
        void calc_k_i(gsl_vector_complex* k_i, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext);
        void calc_k_f(gsl_vector_complex* k_f, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext);


    public:
        ExternalImpulseGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper);
        virtual ~ExternalImpulseGrid();

        gsl_vector_complex* get_l_ext(int idx);
        gsl_vector_complex* get_r_ext(int idx);
        gsl_vector_complex* get_P_ext(int idx);

        gsl_vector_complex* get_p_i(int idx);
        gsl_vector_complex* get_p_f(int idx);
        gsl_vector_complex* get_k_i(int idx);
        gsl_vector_complex* get_k_f(int idx);
};


#endif //NNINTERACTION_EXTERNALIMPULSEGRID_HPP
