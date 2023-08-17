//
// Created by past12am on 8/3/23.
//

#include "../../../include/scattering/processes/QuarkExchange.hpp"
#include "../../../include/operators/ChargeConjugation.hpp"

#include <complex>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>

QuarkExchange::QuarkExchange()
{
    S_p = new QuarkPropagator();
    S_k = new QuarkPropagator();

    D_p = new ScalarDiquarkPropagator();
    D_k = new ScalarDiquarkPropagator();

    Phi_pi = new ScalarQuarkDiquarkAmplitude();
    Phi_ki = new ScalarQuarkDiquarkAmplitude();
    Phi_pf = new ScalarQuarkDiquarkAmplitude();
    Phi_kf = new ScalarQuarkDiquarkAmplitude();
}

QuarkExchange::~QuarkExchange()
{
    delete Phi_kf;
    delete Phi_pf;
    delete Phi_ki;
    delete Phi_pi;

    delete D_k;
    delete D_p;

    delete S_k;
    delete S_p;
}

// Note on notation: p_rp == p_r^' (p_r^prime)
void QuarkExchange::integralKernel(gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                   gsl_vector_complex* p_r, gsl_vector_complex* p_rp,
                                   gsl_vector_complex* p_q, gsl_vector_complex* p_d,
                                   gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                   gsl_vector_complex* k_r, gsl_vector_complex* k_rp,
                                   gsl_vector_complex* k_q, gsl_vector_complex* k_d,
                                   gsl_complex mu2)
{
    gsl_matrix_complex* PhiConj_S_Phi__alpha_delta = gsl_matrix_complex_alloc(4, 4);
    gsl_matrix_complex* PhiConj_S_Phi__gamma_beta = gsl_matrix_complex_alloc(4, 4);



    // matrix_Conj_Phi_pf = ChargeConj(Phi(p_r', p_f))
    gsl_matrix_complex* matrix_Conj_Phi_pf = gsl_matrix_complex_alloc(4, 4);
    Phi_pf->Phi(p_rp, p_f, matrix_Conj_Phi_pf);
    ChargeConjugation::chargeConj(matrix_Conj_Phi_pf);

    // matrix_S_k = S(k_q)
    gsl_matrix_complex* matrix_S_k = gsl_matrix_complex_alloc(4, 4);
    S_k->S(k_q, mu2, matrix_S_k);

    // matrix_Phi_ki = Phi(k_r, k_i)
    gsl_matrix_complex* matrix_Phi_ki = gsl_matrix_complex_alloc(4, 4);
    Phi_ki->Phi(k_r, k_i, matrix_Phi_ki);


    // S_Phi__alpha_delta = S(k_q) Phi(k_r, k_i)
    gsl_matrix_complex S_Phi__alpha_delta;
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_k, matrix_Phi_ki, gsl_complex_rect(0, 0), &S_Phi__alpha_delta);

    // PhiConj_S_Phi__alpha_delta = ChargeConj(Phi(p_r', p_f)) S_Phi__alpha_delta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Phi_pf, &S_Phi__alpha_delta, gsl_complex_rect(0, 0), PhiConj_S_Phi__alpha_delta);

    // PhiConj_S_Phi__alpha_delta = ChargeConj(Phi(p_r', p_f)) S(k_q) Phi(k_r, k_i)





    // matrix_Conj_Phi_kf = ChargeConj(Phi(k_r', k_f))
    gsl_matrix_complex* matrix_Conj_Phi_kf = gsl_matrix_complex_alloc(4, 4);
    Phi_pf->Phi(k_rp, k_f, matrix_Conj_Phi_kf);
    ChargeConjugation::chargeConj(matrix_Conj_Phi_kf);

    // matrix_S_p = S(p_q)
    gsl_matrix_complex* matrix_S_p = gsl_matrix_complex_alloc(4, 4);
    S_k->S(p_q, mu2, matrix_S_p);

    // matrix_Phi_pi = Phi(p_r, p_i)
    gsl_matrix_complex* matrix_Phi_pi = gsl_matrix_complex_alloc(4, 4);
    Phi_ki->Phi(p_r, p_i, matrix_Phi_pi);


    // S_Phi__gamma_beta = S(p_q) Phi(p_r, p_i)
    gsl_matrix_complex S_Phi__gamma_beta;
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_p, matrix_Phi_pi, gsl_complex_rect(0, 0), &S_Phi__gamma_beta);

    // PhiConj_S_Phi__gamma_beta = ChargeConj(Phi(k_r', k_f)) PhiConj_S_Phi__gamma_beta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Phi_kf, &S_Phi__gamma_beta, gsl_complex_rect(0, 0), PhiConj_S_Phi__gamma_beta);




    // scalar_D_p = D(p_d)
    gsl_complex scalar_D_p;
    D_p->D(p_d, &scalar_D_p);

    // scalar_D_k = D(k_d)
    gsl_complex scalar_D_k;
    D_p->D(k_d, &scalar_D_k);
}
