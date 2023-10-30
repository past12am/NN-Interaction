//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_QUARKEXCHANGE_HPP
#define NNINTERACTION_QUARKEXCHANGE_HPP


#include <mutex>
#include "../../qcd/propagators/QuarkPropagator.hpp"
#include "../../qcd/propagators/ScalarDiquarkPropagator.hpp"
#include "../../qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp"
#include "../ScatteringProcess.hpp"
#include "../MomentumLoop.hpp"

class QuarkExchange : public ScatteringProcess
{
    private:
        double eta;

        // Note: One instance per thread
        QuarkPropagator* S_p;
        QuarkPropagator* S_k;

        ScalarDiquarkPropagator* D_p;
        ScalarDiquarkPropagator* D_k;

        ScalarQuarkDiquarkAmplitude* Phi_pi;
        ScalarQuarkDiquarkAmplitude* Phi_ki;
        ScalarQuarkDiquarkAmplitude* Phi_pf;
        ScalarQuarkDiquarkAmplitude* Phi_kf;

        MomentumLoop momentumLoop;

        // Temporary Variables --> Lock when multithreading or create multiple
        gsl_matrix_complex* PhiConj_S_Phi__alpha_delta;
        gsl_matrix_complex* PhiConj_S_Phi__gamma_beta;

        gsl_matrix_complex* S_Phi__alpha_delta;
        gsl_matrix_complex* matrix_Conj_Phi_pf;
        gsl_matrix_complex* matrix_S_k;
        gsl_matrix_complex* matrix_Phi_ki;

        gsl_matrix_complex* S_Phi__gamma_beta;
        gsl_matrix_complex* matrix_Conj_Phi_kf;
        gsl_matrix_complex* matrix_S_p;
        gsl_matrix_complex* matrix_Phi_pi;


        // Temporary Variables for loop impulse --> Lock when multithreading or create multiple
        gsl_vector_complex* k_q;
        gsl_vector_complex* k_d;
        gsl_vector_complex* p_q;
        gsl_vector_complex* p_d;

        gsl_vector_complex* k_r;
        gsl_vector_complex* k_rp;
        gsl_vector_complex* p_r;
        gsl_vector_complex* p_rp;


        gsl_vector_complex* tmp1;


        void calc_k_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* k_q);
        void calc_p_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* p_q);
        void calc_k_d(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* k_d);
        void calc_p_d(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* p_d);

        void calc_k_r(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_r);
        void calc_k_rp(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_rp);
        void calc_p_r(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_r);
        void calc_p_rp(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_rp);



    public:
        QuarkExchange(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper,
                      gsl_complex nucleon_mass, double a, int l2Points, int zPoints, int yPoints, int phiPoints, gsl_complex quarkPropRenormPoint, double eta, int threadIdx);
        virtual ~QuarkExchange();


        void integralKernel(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* K, gsl_vector_complex* P,
                            gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                            gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                            Tensor22<4, 4, 4, 4>* integralKernelTensor) override;

        void integrate(double l2_cutoff);
};


#endif //NNINTERACTION_QUARKEXCHANGE_HPP
