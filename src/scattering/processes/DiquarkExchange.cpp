//
// Created by past12am on 04/01/24.
//

#include <gsl/gsl_blas.h>
#include "../../../include/scattering/processes/DiquarkExchange.hpp"

#include "../../../include/scattering/momentumloops/QuarkExchangeMomentumLoop.hpp"

DiquarkExchange::DiquarkExchange(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double ZCutoffLower, double ZCutoffUpper,
                                 double eta, int k2Points, int zPoints, int yPoints, int phiPoints, int threadIdx) :
        ScatteringProcess(lenX, lenZ, XCutoffLower, XCutoffUpper, ZCutoffLower, ZCutoffUpper, threadIdx),
        eta(eta)
{
    momentumLoop = new QuarkExchangeMomentumLoop(k2Points, zPoints, yPoints, phiPoints);

    lmr_half = gsl_vector_complex_alloc(4);
    lpr_half = gsl_vector_complex_alloc(4);

    S_p = new QuarkPropagator();
    S_k = new QuarkPropagator();

    D_p = new ScalarDiquarkPropagator();
    D_k = new ScalarDiquarkPropagator();

    Gamma_pi = new ScalarQuarkDiquarkAmplitude();
    Gamma_ki = new ScalarQuarkDiquarkAmplitude();
    Gamma_pf = new ScalarQuarkDiquarkAmplitude();
    Gamma_kf = new ScalarQuarkDiquarkAmplitude();


    GammaConj_S_Gamma__alpha_beta = gsl_matrix_complex_alloc(4, 4);
    S_Gamma__alpha_beta = gsl_matrix_complex_alloc(4, 4);

    GammaConj_S_Gamma__gamma_delta = gsl_matrix_complex_alloc(4, 4);
    S_Gamma__gamma_delta = gsl_matrix_complex_alloc(4, 4);

    matrix_Conj_Gamma_kf = gsl_matrix_complex_alloc(4, 4);
    matrix_S_k = gsl_matrix_complex_alloc(4, 4);
    matrix_Gamma_ki = gsl_matrix_complex_alloc(4, 4);

    matrix_Conj_Gamma_pf = gsl_matrix_complex_alloc(4, 4);
    matrix_S_p = gsl_matrix_complex_alloc(4, 4);
    matrix_Gamma_pi = gsl_matrix_complex_alloc(4, 4);


    k_q = gsl_vector_complex_alloc(4);
    k_d = gsl_vector_complex_alloc(4);
    p_q = gsl_vector_complex_alloc(4);
    p_d = gsl_vector_complex_alloc(4);

    k_r = gsl_vector_complex_alloc(4);
    k_rp = gsl_vector_complex_alloc(4);
    p_r = gsl_vector_complex_alloc(4);
    p_rp = gsl_vector_complex_alloc(4);
}

DiquarkExchange::~DiquarkExchange()
{
    gsl_vector_complex_free(lmr_half);
    gsl_vector_complex_free(lpr_half);

    gsl_vector_complex_free(k_r);
    gsl_vector_complex_free(k_rp);
    gsl_vector_complex_free(p_r);
    gsl_vector_complex_free(p_rp);

    gsl_vector_complex_free(k_q);
    gsl_vector_complex_free(k_d);
    gsl_vector_complex_free(p_q);
    gsl_vector_complex_free(p_d);

    gsl_matrix_complex_free(GammaConj_S_Gamma__alpha_beta);
    gsl_matrix_complex_free(GammaConj_S_Gamma__gamma_delta);
    gsl_matrix_complex_free(S_Gamma__alpha_beta);
    gsl_matrix_complex_free(S_Gamma__gamma_delta);
    gsl_matrix_complex_free(matrix_Conj_Gamma_kf);
    gsl_matrix_complex_free(matrix_Conj_Gamma_pf);
    gsl_matrix_complex_free(matrix_Gamma_ki);
    gsl_matrix_complex_free(matrix_Gamma_pi);
    gsl_matrix_complex_free(matrix_S_k);
    gsl_matrix_complex_free(matrix_S_p);

    delete Gamma_kf;
    delete Gamma_pf;
    delete Gamma_ki;
    delete Gamma_pi;

    delete D_k;
    delete D_p;

    delete S_k;
    delete S_p;

    delete static_cast<QuarkExchangeMomentumLoop*>(momentumLoop);
}

void DiquarkExchange::calc_k_q(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *P, gsl_vector_complex *k_q)
{
    // k_q = eta/2 * P
    gsl_vector_complex_memcpy(k_q, P);
    gsl_vector_complex_scale(k_q, gsl_complex_rect(eta * 0.5, 0));

    // k_q = k - (l+r)/2 + eta/2 * P
    gsl_vector_complex_sub(k_q, lpr_half);
    gsl_vector_complex_add(k_q, k);
}

void DiquarkExchange::calc_p_q(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *P, gsl_vector_complex *p_q)
{
    // p_q = eta/2 * P
    gsl_vector_complex_memcpy(p_q, P);
    gsl_vector_complex_scale(p_q, gsl_complex_rect(eta * 0.5, 0));

    // p_q = k + (l+r)/2 + eta/2 * P
    gsl_vector_complex_add(p_q, lpr_half);
    gsl_vector_complex_add(p_q, k);
}

void DiquarkExchange::calc_k_d(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *P, gsl_vector_complex *k_d)
{
    // k_d = (1 - eta)/2 * P
    gsl_vector_complex_memcpy(k_d, P);
    gsl_vector_complex_scale(k_d, gsl_complex_rect((1 - eta) * 0.5, 0));

    // k_d = -k + (l-r)/2 + (1 - eta)/2 * P
    gsl_vector_complex_add(k_d, lmr_half);
    gsl_vector_complex_sub(k_d, k);
}

void DiquarkExchange::calc_p_d(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *P, gsl_vector_complex *p_d)
{
    // p_d = (1 - eta)/2 * P
    gsl_vector_complex_memcpy(p_d, P);
    gsl_vector_complex_scale(p_d, gsl_complex_rect((1 - eta) * 0.5, 0));

    // p_d = -k - (l-r)/2 + (1 - eta)/2 * P
    gsl_vector_complex_sub(p_d, lmr_half);
    gsl_vector_complex_sub(p_d, k);
}

void DiquarkExchange::calc_k_r(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *k_r)
{
    // k_r = eta * r
    gsl_vector_complex_memcpy(k_r, r);
    gsl_vector_complex_scale(k_r, gsl_complex_rect(eta, 0));

    // k_r = k - (l+r)/2 + eta * r
    gsl_vector_complex_sub(k_r, lpr_half);
    gsl_vector_complex_add(k_r, k);
}

void DiquarkExchange::calc_k_rp(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *k_rp)
{
    // k_rp = eta * l
    gsl_vector_complex_memcpy(k_rp, l);
    gsl_vector_complex_scale(k_rp, gsl_complex_rect(eta, 0));

    // k_rp = k - (l+r)/2 + eta * l
    gsl_vector_complex_sub(k_rp, lpr_half);
    gsl_vector_complex_add(k_rp, k);
}

void DiquarkExchange::calc_p_r(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *p_r)
{
    // p_r = -(eta * r)
    gsl_vector_complex_memcpy(p_r, r);
    gsl_vector_complex_scale(p_r, gsl_complex_rect(-eta, 0));

    // p_r = k + (l+r)/2 - eta * r
    gsl_vector_complex_add(p_r, lpr_half);
    gsl_vector_complex_add(p_r, k);
}

void DiquarkExchange::calc_p_rp(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r, gsl_vector_complex *p_rp)
{
    // p_rp = -(eta * l)
    gsl_vector_complex_memcpy(p_rp, l);
    gsl_vector_complex_scale(p_rp, gsl_complex_rect(-eta, 0));

    // p_rp = k + (l+r)/2 - eta * l
    gsl_vector_complex_add(p_rp, lpr_half);
    gsl_vector_complex_add(p_rp, k);
}

void DiquarkExchange::integralKernel(gsl_vector_complex *k, gsl_vector_complex *l, gsl_vector_complex *r,
                                     gsl_vector_complex *P, gsl_vector_complex *p_f, gsl_vector_complex *p_i,
                                     gsl_vector_complex *k_f, gsl_vector_complex *k_i,
                                     Tensor4<4, 4, 4, 4> *integralKernelTensor)
{
    // precalc
    //      lmr_half = (l-r)/2
    gsl_vector_complex_memcpy(lmr_half, l);
    gsl_vector_complex_sub(lmr_half, r);
    gsl_vector_complex_scale(lmr_half, gsl_complex_rect(0.5, 0));

    // precalc
    //      lpr_half = (l+r)/2
    gsl_vector_complex_memcpy(lpr_half, l);
    gsl_vector_complex_add(lpr_half, r);
    gsl_vector_complex_scale(lpr_half, gsl_complex_rect(0.5, 0));


    // Calculate internal impulses
    calc_k_q(k, l, r, P, k_q);
    calc_p_q(k, l, r, P, p_q);
    calc_k_d(k, l, r, P, k_d);
    calc_p_d(k, l, r, P, p_d);

    calc_k_r(k, l, r, k_r);
    calc_k_rp(k, l, r, k_rp);
    calc_p_r(k, l, r, p_r);
    calc_p_rp(k, l, r, p_rp);



    // Calculate tensor-valued integral Kernel

    Gamma_pf->Gamma(p_rp, p_f, true, threadIdx, matrix_Conj_Gamma_pf);
    S_p->S(p_q, matrix_S_p);
    Gamma_pi->Gamma(p_r, p_i, false, threadIdx, matrix_Gamma_pi);

    Gamma_kf->Gamma(k_rp, k_f, true, threadIdx, matrix_Conj_Gamma_kf);
    S_k->S(k_q, matrix_S_k);
    Gamma_ki->Gamma(k_r, k_i, false, threadIdx, matrix_Gamma_ki);


    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, matrix_S_p, matrix_Gamma_pi, GSL_COMPLEX_ZERO, S_Gamma__alpha_beta);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, matrix_Conj_Gamma_pf, S_Gamma__alpha_beta, GSL_COMPLEX_ZERO, GammaConj_S_Gamma__alpha_beta);


    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, matrix_S_k, matrix_Gamma_ki, GSL_COMPLEX_ZERO, S_Gamma__gamma_delta);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, matrix_Conj_Gamma_kf, S_Gamma__gamma_delta, GSL_COMPLEX_ZERO, GammaConj_S_Gamma__gamma_delta);

    gsl_complex scalar_D_p;
    D_p->D(p_d, &scalar_D_p);

    gsl_complex scalar_D_k;
    D_k->D(k_d, &scalar_D_k);

    // Construct M
    for(size_t alpha = 0; alpha < GammaConj_S_Gamma__alpha_beta->size1; alpha++)
    {
        for(size_t beta = 0; beta < GammaConj_S_Gamma__alpha_beta->size2; beta++)
        {
            for(size_t gamma = 0; gamma < GammaConj_S_Gamma__gamma_delta->size1; gamma++)
            {
                for(size_t delta = 0; delta < GammaConj_S_Gamma__gamma_delta->size2; delta++)
                {

                    gsl_complex kernelElement = gsl_complex_mul(gsl_complex_mul(scalar_D_p,
                                                                                   scalar_D_k),
                                                                gsl_complex_mul(gsl_matrix_complex_get(GammaConj_S_Gamma__alpha_beta, alpha, beta),
                                                                                   gsl_matrix_complex_get(GammaConj_S_Gamma__gamma_delta, gamma, delta)));
                    // Set 0 if < 1E-30
                    //if(abs(kernelElement.dat[0]) < 1E-30)
                    //    kernelElement.dat[0] = 0;
                    //if(abs(kernelElement.dat[1]) < 1E-30)
                    //    kernelElement.dat[1] = 0;

                    integralKernelTensor->setElement(alpha, beta, gamma, delta, kernelElement);
                }
            }
        }
    }
}

gsl_complex DiquarkExchange::integrate_process(int basisElemIdx, int externalImpulseIdx, double k2_cutoff)
{
    std::function<gsl_complex(double, double, double, double)> scatteringMatrixIntegrand = [=, this](double k2, double z, double y, double phi) -> gsl_complex {
        return integralKernelWrapper(externalImpulseIdx, basisElemIdx, threadIdx, k2, z, y, phi);
    };

    gsl_complex res = momentumLoop->integrate_4d(scatteringMatrixIntegrand, k2_cutoff);
    return res;
}
