//
// Created by past12am on 8/18/23.
//

#include <cassert>
#include "../../../include/scattering/impulse/ExternalImpulseGrid.hpp"

#include "complex"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_math.h"

ExternalImpulseGrid::ExternalImpulseGrid(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex M_nucleon) :
    ZTauGrid(lenTau, lenZ, tauCutoffLower, tauCutoffUpper, zCutoffLower, zCutoffUpper)
{
    int len = getLength();

    Q = new gsl_vector_complex*[len];
    P = new gsl_vector_complex*[len];
    K = new gsl_vector_complex*[len];

    p_i = new gsl_vector_complex*[len];
    p_f = new gsl_vector_complex*[len];
    k_i = new gsl_vector_complex*[len];
    k_f = new gsl_vector_complex*[len];

    for(int tauIdx = 0; tauIdx < lenTau; tauIdx++)
    {
        for(int zIdx = 0; zIdx < lenZ; zIdx++)
        {
            int i = getGridIdx(tauIdx, zIdx);

            Q[i] = gsl_vector_complex_alloc(4);
            P[i] = gsl_vector_complex_alloc(4);
            K[i] = gsl_vector_complex_alloc(4);
            calc_Q(Q[i], tau[tauIdx], M_nucleon);
            calc_P(P[i], tau[tauIdx], M_nucleon);
            calc_K(K[i], tau[tauIdx], z[zIdx], M_nucleon);

            p_i[i] = gsl_vector_complex_alloc(4);
            p_f[i] = gsl_vector_complex_alloc(4);
            k_i[i] = gsl_vector_complex_alloc(4);
            k_f[i] = gsl_vector_complex_alloc(4);
            calc_p_i(P[i], Q[i], p_i[i]);
            calc_p_f(P[i], Q[i], p_f[i]);
            calc_k_i(K[i], Q[i], k_i[i]);
            calc_k_f(K[i], Q[i], k_f[i]);
        }
    }
}


void ExternalImpulseGrid::calc_p_i(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_i)
{
    gsl_vector_complex_memcpy(p_i, Q);
    gsl_vector_complex_scale(p_i, gsl_complex_rect(-1.0/2.0, 0));
    gsl_vector_complex_add(p_i, P);
}

void ExternalImpulseGrid::calc_p_f(const gsl_vector_complex* P, const gsl_vector_complex* Q, gsl_vector_complex* p_f)
{
    gsl_vector_complex_memcpy(p_f, Q);
    gsl_vector_complex_scale(p_f, gsl_complex_rect(1.0/2.0, 0));
    gsl_vector_complex_add(p_f, P);
}

void ExternalImpulseGrid::calc_k_i(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_i)
{
    gsl_vector_complex_memcpy(k_i, Q);
    gsl_vector_complex_scale(k_i, gsl_complex_rect(1.0/2.0, 0));
    gsl_vector_complex_add(k_i, K);
}

void ExternalImpulseGrid::calc_k_f(const gsl_vector_complex* K, const gsl_vector_complex* Q, gsl_vector_complex* k_f)
{
    gsl_vector_complex_memcpy(k_f, Q);
    gsl_vector_complex_scale(k_f, gsl_complex_rect(-1.0/2.0, 0));
    gsl_vector_complex_add(k_f, K);
}

void ExternalImpulseGrid::calc_Q(gsl_vector_complex* Q, double tau, gsl_complex M_nucleon)
{
    assert(tau > 0);

    gsl_vector_complex_set_zero(Q);
    gsl_vector_complex_set(Q, 3, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul_real(M_nucleon, 2.0 * sqrt(tau));

    gsl_vector_complex_scale(Q, pref);
}

void ExternalImpulseGrid::calc_P(gsl_vector_complex* P, double tau, gsl_complex M_nucleon)
{
    assert((1.0 + tau) > 0);

    gsl_vector_complex_set_zero(P);
    gsl_vector_complex_set(P, 2, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul_real(gsl_complex_mul_imag(M_nucleon, 1), sqrt(1 + tau));

    gsl_vector_complex_scale(P, pref);
}

void ExternalImpulseGrid::calc_K(gsl_vector_complex* K, double tau, double z, gsl_complex M_nucleon)
{
    assert((1.0 + tau) > 0);

    gsl_vector_complex_set_zero(K);
    gsl_vector_complex_set(K, 1, gsl_complex_sqrt(gsl_complex_rect(1 - gsl_pow_2(z), 0)));
    gsl_vector_complex_set(K, 2, gsl_complex_rect(z, 0));

    gsl_complex pref = gsl_complex_mul_real(gsl_complex_mul_imag(M_nucleon, 1), sqrt(1 + tau));

    gsl_vector_complex_scale(K, pref);
}

ExternalImpulseGrid::~ExternalImpulseGrid()
{
    for(int i = 0; i < getLength(); i++)
    {
        gsl_vector_complex_free(Q[i]);
        gsl_vector_complex_free(P[i]);
        gsl_vector_complex_free(K[i]);

        gsl_vector_complex_free(p_i[i]);
        gsl_vector_complex_free(p_f[i]);
        gsl_vector_complex_free(k_i[i]);
        gsl_vector_complex_free(k_f[i]);
    }

    delete []Q;
    delete []P;
    delete []K;

    delete []p_i;
    delete []p_f;
    delete []k_i;
    delete []k_f;
}

gsl_vector_complex* ExternalImpulseGrid::get_Q(int idx)
{
    return Q[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_P(int idx)
{
    return P[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_K(int idx)
{
    return K[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_p_i(int idx)
{
    return p_i[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_p_f(int idx)
{
    return p_f[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_k_i(int idx)
{
    return k_i[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_k_f(int idx)
{
    return k_f[idx];
}


