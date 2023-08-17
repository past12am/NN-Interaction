#include <iostream>
#include "include/scattering/basis/TensorBasis.hpp"

#include <complex>
#include <gsl/gsl_math.h>

void get_Q(gsl_vector_complex* Q, double tau, double m);
void get_P(gsl_vector_complex* P, double tau, double M);
void get_K(gsl_vector_complex* K, double tau, double z, double M);

void calc_p_i(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_i);
void calc_p_f(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_f);
void calc_k_i(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_i);
void calc_k_f(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_f);

int main()
{
    // define impulses
    double tau = 0.5;
    double z = 0;
    double m = 0.004; // GeV
    double M = 0.007; // GeV

    gsl_vector_complex* Q = gsl_vector_complex_alloc(4);
    gsl_vector_complex* P = gsl_vector_complex_alloc(4);
    gsl_vector_complex* K = gsl_vector_complex_alloc(4);
    get_Q(Q, tau, m);
    get_P(P, tau, M);
    get_K(K, tau, z, M);

    gsl_vector_complex* p_i = gsl_vector_complex_alloc(4);
    gsl_vector_complex* p_f = gsl_vector_complex_alloc(4);
    gsl_vector_complex* k_i = gsl_vector_complex_alloc(4);
    gsl_vector_complex* k_f = gsl_vector_complex_alloc(4);
    calc_p_i(P, Q, p_i);
    calc_p_f(P, Q, p_f);
    calc_k_i(K, Q, k_i);
    calc_k_f(K, Q, k_f);





    TensorBasis basis(p_f, p_i, k_f, k_i, P, K);
    std::cout << basis.tau[1] <<std::endl;
    return 0;
}

void calc_p_i(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_i)
{
    gsl_vector_complex_memcpy(p_i, Q);
    gsl_vector_complex_scale(p_i, gsl_complex_rect(-1.0/2.0, 0));
    gsl_vector_complex_add(p_i, P);
}

void calc_p_f(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_f)
{
    gsl_vector_complex_memcpy(p_f, Q);
    gsl_vector_complex_scale(p_f, gsl_complex_rect(1.0/2.0, 0));
    gsl_vector_complex_add(p_f, P);
}

void calc_k_i(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_i)
{
    gsl_vector_complex_memcpy(k_i, Q);
    gsl_vector_complex_scale(k_i, gsl_complex_rect(1.0/2.0, 0));
    gsl_vector_complex_add(k_i, K);
}

void calc_k_f(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_f)
{
    gsl_vector_complex_memcpy(k_f, Q);
    gsl_vector_complex_scale(k_f, gsl_complex_rect(-1.0/2.0, 0));
    gsl_vector_complex_add(k_f, K);
}

void get_Q(gsl_vector_complex* Q, double tau, double m)
{
    gsl_vector_complex_set_zero(Q);
    gsl_vector_complex_set(Q, 3, gsl_complex_rect(1, 0));

    gsl_complex sqrt_tau = gsl_complex_sqrt(gsl_complex_rect(tau, 0));
    double pref = 2.0 * m;

    gsl_vector_complex_scale(Q, gsl_complex_mul_real(sqrt_tau, pref));
}

void get_P(gsl_vector_complex* P, double tau, double M)
{
    gsl_vector_complex_set_zero(P);
    gsl_vector_complex_set(P, 2, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul(gsl_complex_rect(0, M), gsl_complex_sqrt(gsl_complex_rect(1 + tau, 0)));

    gsl_vector_complex_scale(P, pref);
}

void get_K(gsl_vector_complex* K, double tau, double z, double M)
{
    gsl_vector_complex_set_zero(K);
    gsl_vector_complex_set(K, 1, gsl_complex_sqrt(gsl_complex_rect(1 - gsl_pow_2(z), 0)));
    gsl_vector_complex_set(K, 2, gsl_complex_rect(z, 0));

    gsl_complex pref = gsl_complex_mul(gsl_complex_rect(0, M), gsl_complex_sqrt(gsl_complex_rect(1 + tau, 0)));

    gsl_vector_complex_scale(K, pref);
}