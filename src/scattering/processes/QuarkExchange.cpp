//
// Created by past12am on 8/3/23.
//

#include "../../../include/scattering/processes/QuarkExchange.hpp"
#include "../../../include/operators/ChargeConjugation.hpp"
#include "../../../include/utils/print/PrintGSLElements.hpp"

#include <complex>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_integration.h>

#include <iostream>


QuarkExchange::QuarkExchange(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double ZCutoffLower, double ZCutoffUpper,
                             gsl_complex nucleon_mass, double eta, int k2Points, int zPoints, int yPoints, int phiPoints, int threadIdx) :
                                        ScatteringProcess(lenX, lenZ, XCutoffLower, XCutoffUpper, ZCutoffLower, ZCutoffUpper, nucleon_mass, threadIdx),
                                        eta(eta),
                                        momentumLoop(k2Points, zPoints, yPoints, phiPoints)
{
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

    GammaConj_S_Gamma__alpha_delta = gsl_matrix_complex_alloc(4, 4);
    GammaConj_S_Gamma__gamma_beta = gsl_matrix_complex_alloc(4, 4);

    S_Gamma__alpha_delta = gsl_matrix_complex_alloc(4, 4);
    matrix_Conj_Gamma_pf = gsl_matrix_complex_alloc(4, 4);
    matrix_S_k = gsl_matrix_complex_alloc(4, 4);
    matrix_Gamma_ki = gsl_matrix_complex_alloc(4, 4);

    S_Gamma__gamma_beta = gsl_matrix_complex_alloc(4, 4);
    matrix_Conj_Gamma_kf = gsl_matrix_complex_alloc(4, 4);
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

QuarkExchange::~QuarkExchange()
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

    gsl_matrix_complex_free(GammaConj_S_Gamma__alpha_delta);
    gsl_matrix_complex_free(GammaConj_S_Gamma__gamma_beta);
    gsl_matrix_complex_free(S_Gamma__alpha_delta);
    gsl_matrix_complex_free(S_Gamma__gamma_beta);
    gsl_matrix_complex_free(matrix_Conj_Gamma_pf);
    gsl_matrix_complex_free(matrix_S_k);
    gsl_matrix_complex_free(matrix_Gamma_ki);
    gsl_matrix_complex_free(matrix_Conj_Gamma_kf);
    gsl_matrix_complex_free(matrix_S_p);
    gsl_matrix_complex_free(matrix_Gamma_pi);

    delete Gamma_kf;
    delete Gamma_pf;
    delete Gamma_ki;
    delete Gamma_pi;

    delete D_k;
    delete D_p;

    delete S_k;
    delete S_p;
}

void QuarkExchange::integrate(double k2_cutoff)
{
    int num_progress_char = 100;
    int progress = 0;
    int total = tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength();

    double avg_time = 0;
    clock_t clock_at_start = clock();
    clock_t clock_at_end = clock();

    std::cout << "Thread " << threadIdx << " calculates " << tensorBasis.getTensorBasisElementCount() * externalImpulseGrid.getLength() << " grid points" << std::endl;

    for(int basisElemIdx = 0; basisElemIdx < tensorBasis.getTensorBasisElementCount(); basisElemIdx++)
    {
        // Integrate each Scattering Matrix element for each choice of external Impulse
        for (int externalImpulseIdx = 0; externalImpulseIdx < externalImpulseGrid.getLength(); externalImpulseIdx++)
        {
            progress = basisElemIdx * externalImpulseGrid.getLength() + externalImpulseIdx;
            avg_time = (clock_at_end - clock_at_start)/(progress + 1);

            std::cout << "Thread " << threadIdx << "-->   Basis Element [" << std::string(((double) progress/total) * num_progress_char, '#') << std::string((1.0 - (double) progress/total) * num_progress_char, '-').c_str() << "]    --> "
                << ((clock_at_end - clock_at_start)/CLOCKS_PER_SEC)/60 << " of " << ((avg_time * total)/CLOCKS_PER_SEC)/60 << "\t" << std::flush;



            std::function<gsl_complex(double, double, double, double)> scatteringMatrixIntegrand = [=, this](double k2, double z, double y, double phi) -> gsl_complex {
                return integralKernelWrapper(externalImpulseIdx, basisElemIdx, threadIdx, k2, z, y, phi);
            };

            gsl_complex res = momentumLoop.k2Integral(scatteringMatrixIntegrand, 0, k2_cutoff);
            res = gsl_complex_mul_real(res, 1.0/pow(2.0 * std::numbers::pi, 4) * 0.5);

            //assert(GSL_IMAG(res) == 0);

            scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)] = res;

            std::cout << "Basis[" << basisElemIdx << "], basisIdx=" << externalImpulseIdx << ": " << GSL_REAL(res) << " + i " << GSL_IMAG(res) << std::endl;

            clock_at_end = clock();
        }
    }
}

// Note on notation: p_rp == p_r^' (p_r^prime)
void QuarkExchange::integralKernel(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P,
                                   gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                   gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                   Tensor4<4, 4, 4, 4>* integralKernelTensor)
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


    calc_k_q(k, l, r, P, k_q);
    calc_p_q(k, l, r, P, p_q);
    calc_k_d(k, l, r, P, k_d);
    calc_p_d(k, l, r, P, p_d);

    calc_k_r(k, l, r, k_r);
    calc_k_rp(k, l, r, k_rp);
    calc_p_r(k, l, r, p_r);
    calc_p_rp(k, l, r, p_rp);


    // matrix_Conj_Gamma_pf = ChargeConj(Gamma(p_r', p_f))
    Gamma_pf->Gamma(p_rp, p_f, true, threadIdx, matrix_Conj_Gamma_pf);

    // matrix_S_k = S(k_q)
    S_k->S(k_q,  matrix_S_k);

    // matrix_Gamma_ki = Gamma(k_r, k_i)
    Gamma_ki->Gamma(k_r, k_i, false, threadIdx, matrix_Gamma_ki);


    // S_Gamma__alpha_delta = S(k_q) Gamma(k_r, k_i)
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_k, matrix_Gamma_ki, gsl_complex_rect(0, 0), S_Gamma__alpha_delta);

    // GammaConj_S_Gamma__alpha_delta = ChargeConj(Gamma(p_r', p_f)) S_Gamma__alpha_delta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Gamma_pf, S_Gamma__alpha_delta, gsl_complex_rect(0, 0), GammaConj_S_Gamma__alpha_delta);

    // GammaConj_S_Gamma__alpha_delta = ChargeConj(Gamma(p_r', p_f)) S(k_q) Gamma(k_r, k_i)




    // matrix_Conj_Gamma_kf = ChargeConj(Gamma(k_r', k_f))
    Gamma_pf->Gamma(k_rp, k_f, true, threadIdx, matrix_Conj_Gamma_kf);

    // matrix_S_p = S(p_q)
    S_k->S(p_q,  matrix_S_p);

    // matrix_Gamma_pi = Gamma(p_r, p_i)
    Gamma_ki->Gamma(p_r, p_i, false, threadIdx, matrix_Gamma_pi);


    // S_Gamma__gamma_beta = S(p_q) Gamma(p_r, p_i)
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_p, matrix_Gamma_pi, gsl_complex_rect(0, 0), S_Gamma__gamma_beta);

    // GammaConj_S_Gamma__gamma_beta = ChargeConj(Gamma(k_r', k_f)) GammaConj_S_Gamma__gamma_beta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Gamma_kf, S_Gamma__gamma_beta, gsl_complex_rect(0, 0), GammaConj_S_Gamma__gamma_beta);

    // GammaConj_S_Gamma__gamma_beta = ChargeConj(Gamma(k_r', k_f)) S(p_q) Gamma(p_r, p_i)



    // scalar_D_p = D(p_d)
    gsl_complex scalar_D_p;
    D_p->D(p_d, &scalar_D_p);

    // scalar_D_k = D(k_d)
    gsl_complex scalar_D_k;
    D_p->D(k_d, &scalar_D_k);



    // Construct M
    for(size_t alpha = 0; alpha < GammaConj_S_Gamma__alpha_delta->size1; alpha++)
    {
        for(size_t delta = 0; delta < GammaConj_S_Gamma__alpha_delta->size2; delta++)
        {
            for(size_t gamma = 0; gamma < GammaConj_S_Gamma__gamma_beta->size1; gamma++)
            {
                for(size_t beta = 0; beta < GammaConj_S_Gamma__gamma_beta->size2; beta++)
                {
                    gsl_complex kernelElement = gsl_complex_mul(gsl_complex_mul(scalar_D_p,
                                                                                   scalar_D_k),
                                                                gsl_complex_mul(gsl_matrix_complex_get(GammaConj_S_Gamma__alpha_delta, alpha, delta),
                                                                                   gsl_matrix_complex_get(GammaConj_S_Gamma__gamma_beta, gamma, beta)));
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

// TODO possible performance increase --> (l+-r)/2 pre-calc for all used internal momenta
void QuarkExchange::calc_k_q(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* k_q)
{
    // k_q = eta/2 * P
    gsl_vector_complex_memcpy(k_q, P);
    gsl_vector_complex_scale(k_q, gsl_complex_rect(eta * 0.5, 0));

    // k_q = k + (l-r)/2 + eta/2 * P
    gsl_vector_complex_add(k_q, lmr_half);
    gsl_vector_complex_add(k_q, k);
}

void QuarkExchange::calc_p_q(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* p_q)
{
    // p_q = eta/2 * P
    gsl_vector_complex_memcpy(p_q, P);
    gsl_vector_complex_scale(p_q, gsl_complex_rect(eta * 0.5, 0));

    // p_q = k - (l-r)/2 + eta/2 * P
    gsl_vector_complex_sub(p_q, lmr_half);
    gsl_vector_complex_add(p_q, k);
}

void QuarkExchange::calc_k_d(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* k_d)
{
    // k_d = (1 - eta)/2 * P
    gsl_vector_complex_memcpy(k_d, P);
    gsl_vector_complex_scale(k_d, gsl_complex_rect((1 - eta) * 0.5, 0));

    // k_d = -k - (l+r)/2 + (1 - eta)/2 * P
    gsl_vector_complex_sub(k_d, lpr_half);
    gsl_vector_complex_sub(k_d, k);
}

void QuarkExchange::calc_p_d(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P, gsl_vector_complex* p_d)
{
    // p_d = (1 - eta)/2 * P
    gsl_vector_complex_memcpy(p_d, P);
    gsl_vector_complex_scale(p_d, gsl_complex_rect((1 - eta) * 0.5, 0));

    // p_d = -k + (l+r)/2 + (1 - eta)/2 * P
    gsl_vector_complex_add(p_d, lpr_half);
    gsl_vector_complex_sub(p_d, k);
}

void QuarkExchange::calc_k_r(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* k_r)
{
    //tmp1_mutex.lock();

    // k_r = eta * r
    gsl_vector_complex_memcpy(k_r, r);
    gsl_vector_complex_scale(k_r, gsl_complex_rect(eta, 0));

    // k_r = k + (l-r)/2 + eta * r
    gsl_vector_complex_add(k_r, lmr_half);
    gsl_vector_complex_add(k_r, k);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_k_rp(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* k_rp)
{
    //tmp1_mutex.lock();

    // k_rp = eta * l
    gsl_vector_complex_memcpy(k_rp, l);
    gsl_vector_complex_scale(k_rp, gsl_complex_rect(eta, 0));

    // k_rp = k - (l-r)/2 + eta * l
    gsl_vector_complex_sub(k_rp, lmr_half);
    gsl_vector_complex_add(k_rp, k);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_p_r(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* p_r)
{
    //tmp1_mutex.lock();

    // p_r = -(eta * l)
    gsl_vector_complex_memcpy(p_r, l);
    gsl_vector_complex_scale(p_r, gsl_complex_rect(-eta, 0));

    // p_r = k + (l-r)/2 - eta * l
    gsl_vector_complex_add(p_r, lmr_half);
    gsl_vector_complex_add(p_r, k);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_p_rp(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* p_rp)
{
    //tmp1_mutex.lock();

    // p_rp = -(eta * r)
    gsl_vector_complex_memcpy(p_rp, r);
    gsl_vector_complex_scale(p_rp, gsl_complex_rect(-eta, 0));

    // p_rp = k - (l-r)/2 - eta * r
    gsl_vector_complex_sub(p_rp, lmr_half);
    gsl_vector_complex_add(p_rp, k);

    //tmp1_mutex.unlock();
}
