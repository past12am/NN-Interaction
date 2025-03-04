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

        ScalarQuarkDiquarkAmplitude* Gamma_pi;
        ScalarQuarkDiquarkAmplitude* Gamma_ki;
        ScalarQuarkDiquarkAmplitude* Gamma_pf;
        ScalarQuarkDiquarkAmplitude* Gamma_kf;

        MomentumLoop momentumLoop;

        // Temporary Variables --> Lock when multithreading or create multiple
        gsl_matrix_complex* GammaConj_S_Gamma__alpha_delta;
        gsl_matrix_complex* GammaConj_S_Gamma__gamma_beta;

        gsl_matrix_complex* S_Gamma__alpha_delta;
        gsl_matrix_complex* matrix_Conj_Gamma_pf;
        gsl_matrix_complex* matrix_S_k;
        gsl_matrix_complex* matrix_Gamma_ki;

        gsl_matrix_complex* S_Gamma__gamma_beta;
        gsl_matrix_complex* matrix_Conj_Gamma_kf;
        gsl_matrix_complex* matrix_S_p;
        gsl_matrix_complex* matrix_Gamma_pi;


        // Temporary Variables for loop impulse --> Lock when multithreading or create multiple
        gsl_vector_complex* k_q;
        gsl_vector_complex* k_d;
        gsl_vector_complex* p_q;
        gsl_vector_complex* p_d;

        gsl_vector_complex* k_r;
        gsl_vector_complex* k_rp;
        gsl_vector_complex* p_r;
        gsl_vector_complex* p_rp;


        gsl_vector_complex* lmr_half;
        gsl_vector_complex* lpr_half;


        void calc_k_q(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* k_q);
        void calc_p_q(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* p_q);
        void calc_k_d(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* k_d);
        void calc_p_d(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* p_d);

        void calc_k_r(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* k_r);
        void calc_k_rp(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* k_rp);
        void calc_p_r(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* p_r);
        void calc_p_rp(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* p_rp);



    public:
        QuarkExchange(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double ZCutoffLower, double ZCutoffUpper,
                      gsl_complex nucleon_mass, double eta, int k2Points, int zPoints, int yPoints, int phiPoints, int threadIdx);
        ~QuarkExchange() override;


        void integralKernel(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P,
                            gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                            gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                            Tensor4<4, 4, 4, 4>* integralKernelTensor) override;

        gsl_complex integrate_process(int basisElemIdx, int externalImpulseIdx, double k2_cutoff) override;
};


#endif //NNINTERACTION_QUARKEXCHANGE_HPP
