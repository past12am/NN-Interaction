//
// Created by past12am on 8/3/23.
//

#include "../../../include/scattering/processes/QuarkExchange.hpp"
#include "../../../include/operators/ChargeConjugation.hpp"

#include <complex>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_integration.h>

#include <iostream>


QuarkExchange::QuarkExchange(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, gsl_complex a,
                             int l2Points, int zPoints, int yPoints, int phiPoints, gsl_complex quarkPropRenormPoint, double eta, int threadIdx) :
                                        eta(eta),
                                        ScatteringProcess(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper, nucleon_mass, a, threadIdx),
                                        momentumLoop(l2Points, zPoints, yPoints, phiPoints)
{
    tmp1 = gsl_vector_complex_alloc(4);

    S_p = new QuarkPropagator(quarkPropRenormPoint);
    S_k = new QuarkPropagator(quarkPropRenormPoint);

    D_p = new ScalarDiquarkPropagator();
    D_k = new ScalarDiquarkPropagator();

    Phi_pi = new ScalarQuarkDiquarkAmplitude();
    Phi_ki = new ScalarQuarkDiquarkAmplitude();
    Phi_pf = new ScalarQuarkDiquarkAmplitude();
    Phi_kf = new ScalarQuarkDiquarkAmplitude();

    PhiConj_S_Phi__alpha_delta = gsl_matrix_complex_alloc(4, 4);
    PhiConj_S_Phi__gamma_beta = gsl_matrix_complex_alloc(4, 4);

    S_Phi__alpha_delta = gsl_matrix_complex_alloc(4, 4);
    matrix_Conj_Phi_pf = gsl_matrix_complex_alloc(4, 4);
    matrix_S_k = gsl_matrix_complex_alloc(4, 4);
    matrix_Phi_ki = gsl_matrix_complex_alloc(4, 4);

    S_Phi__gamma_beta = gsl_matrix_complex_alloc(4, 4);
    matrix_Conj_Phi_kf = gsl_matrix_complex_alloc(4, 4);
    matrix_S_p = gsl_matrix_complex_alloc(4, 4);
    matrix_Phi_pi = gsl_matrix_complex_alloc(4, 4);


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
    gsl_vector_complex_free(tmp1);

    gsl_vector_complex_free(k_r);
    gsl_vector_complex_free(k_rp);
    gsl_vector_complex_free(p_r);
    gsl_vector_complex_free(p_rp);

    gsl_vector_complex_free(k_q);
    gsl_vector_complex_free(k_d);
    gsl_vector_complex_free(p_q);
    gsl_vector_complex_free(p_d);

    gsl_matrix_complex_free(PhiConj_S_Phi__alpha_delta);
    gsl_matrix_complex_free(PhiConj_S_Phi__gamma_beta);
    gsl_matrix_complex_free(S_Phi__alpha_delta);
    gsl_matrix_complex_free(S_Phi__gamma_beta);
    gsl_matrix_complex_free(matrix_Conj_Phi_pf);
    gsl_matrix_complex_free(matrix_S_k);
    gsl_matrix_complex_free(matrix_Phi_ki);
    gsl_matrix_complex_free(matrix_Conj_Phi_kf);
    gsl_matrix_complex_free(matrix_S_p);
    gsl_matrix_complex_free(matrix_Phi_pi);

    delete Phi_kf;
    delete Phi_pf;
    delete Phi_ki;
    delete Phi_pi;

    delete D_k;
    delete D_p;

    delete S_k;
    delete S_p;
}

void QuarkExchange::integrate(double l2_cutoff)
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

            std::function<gsl_complex(double, double, double, double)> scatteringMatrixIntegrand = [=, this](double l2, double z, double y, double phi) -> gsl_complex {
                return integralKernelWrapper(externalImpulseIdx, basisElemIdx, threadIdx, l2, z, y, phi);
            };

            gsl_complex res = momentumLoop.l2Integral(scatteringMatrixIntegrand, 0, l2_cutoff);
            res = gsl_complex_mul_real(res, 1.0/pow(2.0 * std::numbers::pi, 4) * 0.5);

            //assert(GSL_IMAG(res) == 0);

            scattering_amplitude_basis_projected[calcScatteringAmpIdx(basisElemIdx, externalImpulseIdx)] = res;

            std::cout << "tau[" << basisElemIdx << "], basisIdx=" << externalImpulseIdx << ": " << GSL_REAL(res) << " + i " << GSL_IMAG(res) << std::endl;

            clock_at_end = clock();
        }
    }
}

// Note on notation: p_rp == p_r^' (p_r^prime)
void QuarkExchange::integralKernel(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* K, gsl_vector_complex* P,
                                   gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                   gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                   Tensor4<4, 4, 4, 4>* integralKernelTensor)
{
    calc_k_q(l, Q, k_q);
    calc_p_q(l, Q, p_q);
    calc_k_d(l, K, k_d);
    calc_p_d(l, P, p_d);

    calc_k_r(l, K, Q, eta, k_r);
    calc_k_rp(l, K, Q, eta, k_rp);
    calc_p_r(l, P, Q, eta, p_r);
    calc_p_rp(l, P, Q, eta, p_rp);


    // matrix_Conj_Phi_pf = ChargeConj(Phi(p_r', p_f))
    Phi_pf->Phi(p_rp, p_f, matrix_Conj_Phi_pf);
    ChargeConjugation::chargeConj(matrix_Conj_Phi_pf, threadIdx);

    // matrix_S_k = S(k_q)
    S_k->S(k_q,  matrix_S_k);

    // matrix_Phi_ki = Phi(k_r, k_i)
    Phi_ki->Phi(k_r, k_i, matrix_Phi_ki);


    // S_Phi__alpha_delta = S(k_q) Phi(k_r, k_i)
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_k, matrix_Phi_ki, gsl_complex_rect(0, 0), S_Phi__alpha_delta);

    // PhiConj_S_Phi__alpha_delta = ChargeConj(Phi(p_r', p_f)) S_Phi__alpha_delta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Phi_pf, S_Phi__alpha_delta, gsl_complex_rect(0, 0), PhiConj_S_Phi__alpha_delta);

    // PhiConj_S_Phi__alpha_delta = ChargeConj(Phi(p_r', p_f)) S(k_q) Phi(k_r, k_i)





    // matrix_Conj_Phi_kf = ChargeConj(Phi(k_r', k_f))
    Phi_pf->Phi(k_rp, k_f, matrix_Conj_Phi_kf);
    ChargeConjugation::chargeConj(matrix_Conj_Phi_kf, threadIdx);

    // matrix_S_p = S(p_q)
    S_k->S(p_q,  matrix_S_p);

    // matrix_Phi_pi = Phi(p_r, p_i)
    Phi_ki->Phi(p_r, p_i, matrix_Phi_pi);


    // S_Phi__gamma_beta = S(p_q) Phi(p_r, p_i)
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_S_p, matrix_Phi_pi, gsl_complex_rect(0, 0), S_Phi__gamma_beta);

    // PhiConj_S_Phi__gamma_beta = ChargeConj(Phi(k_r', k_f)) PhiConj_S_Phi__gamma_beta
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1, 0), matrix_Conj_Phi_kf, S_Phi__gamma_beta, gsl_complex_rect(0, 0), PhiConj_S_Phi__gamma_beta);




    // scalar_D_p = D(p_d)
    gsl_complex scalar_D_p;
    D_p->D(p_d, &scalar_D_p);

    // scalar_D_k = D(k_d)
    gsl_complex scalar_D_k;
    D_p->D(k_d, &scalar_D_k);



    // Construct M
    for(int alpha = 0; alpha < PhiConj_S_Phi__alpha_delta->size1; alpha++)
    {
        for(int beta = 0; beta < PhiConj_S_Phi__gamma_beta->size2; beta++)
        {
            for(int gamma = 0; gamma < PhiConj_S_Phi__gamma_beta->size1; gamma++)
            {
                for(int delta = 0; delta < PhiConj_S_Phi__alpha_delta->size2; delta++)
                {
                    gsl_complex kernelElement = gsl_complex_mul(scalar_D_p, scalar_D_k);
                    kernelElement = gsl_complex_mul(kernelElement, gsl_complex_mul(gsl_matrix_complex_get(PhiConj_S_Phi__alpha_delta, alpha, delta),
                                                                                         gsl_matrix_complex_get(PhiConj_S_Phi__gamma_beta,  gamma, beta)));

                    //std::cout << kernelElement.dat[0] << "+i " << kernelElement.dat[1] << std::endl;
                    integralKernelTensor->setElement(alpha, beta, gamma, delta, kernelElement);
                }
            }
        }
    }
}

void QuarkExchange::calc_k_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* k_q)
{
    gsl_vector_complex_memcpy(k_q, Q);
    gsl_vector_complex_scale(k_q, gsl_complex_rect(1.0/2.0, 0));
    gsl_vector_complex_add(k_q, l);
}

void QuarkExchange::calc_p_q(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* p_q)
{
    gsl_vector_complex_memcpy(p_q, Q);
    gsl_vector_complex_scale(p_q, gsl_complex_rect(-1.0/2.0, 0));
    gsl_vector_complex_add(p_q, l);
}

void QuarkExchange::calc_k_d(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* k_d)
{
    gsl_vector_complex_memcpy(k_d, K);
    gsl_vector_complex_sub(k_d, l);
}

void QuarkExchange::calc_p_d(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* p_d)
{
    gsl_vector_complex_memcpy(p_d, P);
    gsl_vector_complex_sub(p_d, l);
}

void QuarkExchange::calc_k_r(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_r)
{
    //tmp1_mutex.lock();

    gsl_vector_complex_memcpy(tmp1, K);
    gsl_vector_complex_scale(tmp1, gsl_complex_rect(-eta, 0));

    gsl_vector_complex_memcpy(k_r, Q);
    gsl_vector_complex_scale(k_r, gsl_complex_rect((1.0 - eta)/2.0, 0));

    gsl_vector_complex_add(k_r, tmp1);
    gsl_vector_complex_add(k_r, l);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_k_rp(gsl_vector_complex* l, gsl_vector_complex* K, gsl_vector_complex* Q, double eta, gsl_vector_complex* k_rp)
{
    //tmp1_mutex.lock();

    gsl_vector_complex_memcpy(tmp1, K);
    gsl_vector_complex_scale(tmp1, gsl_complex_rect(-eta, 0));

    gsl_vector_complex_memcpy(k_rp, Q);
    gsl_vector_complex_scale(k_rp, gsl_complex_rect(-(1.0 - eta)/2.0, 0));

    gsl_vector_complex_add(k_rp, tmp1);
    gsl_vector_complex_add(k_rp, l);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_p_r(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_r)
{
    //tmp1_mutex.lock();

    gsl_vector_complex_memcpy(tmp1, P);
    gsl_vector_complex_scale(tmp1, gsl_complex_rect(-eta, 0));

    gsl_vector_complex_memcpy(p_r, Q);
    gsl_vector_complex_scale(p_r, gsl_complex_rect(-(1.0 - eta)/2.0, 0));

    gsl_vector_complex_add(p_r, tmp1);
    gsl_vector_complex_add(p_r, l);

    //tmp1_mutex.unlock();
}

void QuarkExchange::calc_p_rp(gsl_vector_complex* l, gsl_vector_complex* P, gsl_vector_complex* Q, double eta, gsl_vector_complex* p_rp)
{
    //tmp1_mutex.lock();

    gsl_vector_complex_memcpy(tmp1, P);
    gsl_vector_complex_scale(tmp1, gsl_complex_rect(-eta, 0));

    gsl_vector_complex_memcpy(p_rp, Q);
    gsl_vector_complex_scale(p_rp, gsl_complex_rect((1.0 - eta)/2.0, 0));

    gsl_vector_complex_add(p_rp, tmp1);
    gsl_vector_complex_add(p_rp, l);

    //tmp1_mutex.unlock();
}
