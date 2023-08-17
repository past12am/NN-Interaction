//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_QUARKEXCHANGE_HPP
#define NNINTERACTION_QUARKEXCHANGE_HPP


#include "../../propagators/QuarkPropagator.hpp"
#include "../../propagators/ScalarDiquarkPropagator.hpp"
#include "../../amplitudes/ScalarQuarkDiquarkAmplitude.hpp"
#include "../ScatteringProcess.hpp"

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

    public:
        QuarkExchange();
        virtual ~QuarkExchange();

        void integralKernel(gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                            gsl_vector_complex* p_r, gsl_vector_complex* p_rp,
                            gsl_vector_complex* p_q, gsl_vector_complex* p_d,
                            gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                            gsl_vector_complex* k_r, gsl_vector_complex* k_rp,
                            gsl_vector_complex* k_q, gsl_vector_complex* k_d,
                            gsl_complex mu2) override;
};


#endif //NNINTERACTION_QUARKEXCHANGE_HPP
