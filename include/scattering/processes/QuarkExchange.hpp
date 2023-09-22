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
        QuarkPropagator* S_p;
        QuarkPropagator* S_k;

        ScalarDiquarkPropagator* D_p;
        ScalarDiquarkPropagator* D_k;

        ScalarQuarkDiquarkAmplitude* Phi_pi;
        ScalarQuarkDiquarkAmplitude* Phi_ki;
        ScalarQuarkDiquarkAmplitude* Phi_pf;
        ScalarQuarkDiquarkAmplitude* Phi_kf;

        MomentumLoop momentumLoop;

        // Temporary Variables --> Lock when multithreading
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


        // Temporary Variables for loop impulse
        gsl_vector_complex* k_q;
        gsl_vector_complex* k_d;
        gsl_vector_complex* p_q;
        gsl_vector_complex* p_d;

        gsl_vector_complex* k_r;
        gsl_vector_complex* k_rp;
        gsl_vector_complex* p_r;
        gsl_vector_complex* p_rp;


        // Shared temporary variables
        static std::mutex tmp1_mutex;
        static gsl_vector_complex* tmp1;


        static void calc_k_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* k_q);
        static void calc_p_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* p_q);
        static void calc_k_d(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* k_d);
        static void calc_p_d(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* p_d);

        static void calc_k_r(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_r);
        static void calc_k_rp(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_rp);
        static void calc_p_r(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_r);
        static void calc_p_rp(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_rp);



    public:
        QuarkExchange(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper,
                      gsl_complex M_nucleon, int l2Points, int zPoints, int yPoints, int phiPoints, gsl_complex quarkPropRenormPoint);
        virtual ~QuarkExchange();

        gsl_complex integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, double l2, double z, double y, double phi);

        void integralKernel(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* K, gsl_vector_complex* P,
                            gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                            gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                            Tensor4<4, 4, 4, 4>* integralKernelTensor) override;

        void integrate();
};


#endif //NNINTERACTION_QUARKEXCHANGE_HPP
