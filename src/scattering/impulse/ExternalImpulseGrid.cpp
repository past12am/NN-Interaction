//
// Created by past12am on 8/18/23.
//

#include <cassert>
#include <gsl/gsl_blas.h>
#include "../../../include/scattering/impulse/ExternalImpulseGrid.hpp"

#include "complex"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_math.h"

ExternalImpulseGrid::ExternalImpulseGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, double a) :
        nucleon_mass(nucleon_mass), ZXGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper)
{
    int len = getLength();

    k_ext = new gsl_vector_complex*[len];
    p_ext = new gsl_vector_complex*[len];
    q_ext = new gsl_vector_complex*[len];

    q_ext_timelike = new gsl_vector_complex*[len];

    Q = new gsl_vector_complex*[len];
    P = new gsl_vector_complex*[len];
    K = new gsl_vector_complex*[len];

    P_timelike = new gsl_vector_complex*[len];

    p_i = new gsl_vector_complex*[len];
    p_f = new gsl_vector_complex*[len];
    k_i = new gsl_vector_complex*[len];
    k_f = new gsl_vector_complex*[len];

    p_i_timelike = new gsl_vector_complex*[len];
    p_f_timelike = new gsl_vector_complex*[len];
    k_i_timelike = new gsl_vector_complex*[len];
    k_f_timelike = new gsl_vector_complex*[len];

    for(int XIdx = 0; XIdx < lenX; XIdx++)
    {
        for(int zIdx = 0; zIdx < lenZ; zIdx++)
        {
            int i = getGridIdx(XIdx, zIdx);

            k_ext[i] = gsl_vector_complex_alloc(4);
            p_ext[i] = gsl_vector_complex_alloc(4);
            q_ext[i] = gsl_vector_complex_alloc(4);
            calc_k_ext(k_ext[i], X[XIdx], nucleon_mass, z[zIdx]);
            calc_p_ext(p_ext[i], X[XIdx], nucleon_mass);
            calc_q_ext(q_ext[i], X[XIdx], nucleon_mass, a);

            q_ext_timelike[i] = gsl_vector_complex_alloc(4);
            calc_q_ext_timelike(q_ext_timelike[i], X[XIdx], nucleon_mass);

            Q[i] = gsl_vector_complex_alloc(4);
            P[i] = gsl_vector_complex_alloc(4);
            K[i] = gsl_vector_complex_alloc(4);
            calc_Q(Q[i], k_ext[i], p_ext[i], q_ext[i]);
            calc_P(P[i], k_ext[i], p_ext[i], q_ext[i]);
            calc_K(K[i], k_ext[i], p_ext[i], q_ext[i]);

            P_timelike[i] = gsl_vector_complex_alloc(4);
            calc_P(P_timelike[i], k_ext[i], p_ext[i], q_ext_timelike[i]);

            p_i[i] = gsl_vector_complex_alloc(4);
            p_f[i] = gsl_vector_complex_alloc(4);
            k_i[i] = gsl_vector_complex_alloc(4);
            k_f[i] = gsl_vector_complex_alloc(4);
            calc_p_i(p_i[i], k_ext[i], p_ext[i], q_ext[i]);
            calc_p_f(p_f[i], k_ext[i], p_ext[i], q_ext[i]);
            calc_k_i(k_i[i], k_ext[i], p_ext[i], q_ext[i]);
            calc_k_f(k_f[i], k_ext[i], p_ext[i], q_ext[i]);

            p_i_timelike[i] = gsl_vector_complex_alloc(4);
            p_f_timelike[i] = gsl_vector_complex_alloc(4);
            k_i_timelike[i] = gsl_vector_complex_alloc(4);
            k_f_timelike[i] = gsl_vector_complex_alloc(4);
            calc_p_i(p_i_timelike[i], k_ext[i], p_ext[i], q_ext_timelike[i]);
            calc_p_f(p_f_timelike[i], k_ext[i], p_ext[i], q_ext_timelike[i]);
            calc_k_i(k_i_timelike[i], k_ext[i], p_ext[i], q_ext_timelike[i]);
            calc_k_f(k_f_timelike[i], k_ext[i], p_ext[i], q_ext_timelike[i]);
        }
    }
}

ExternalImpulseGrid::~ExternalImpulseGrid()
{
    for(int i = 0; i < getLength(); i++)
    {
        gsl_vector_complex_free(Q[i]);
        gsl_vector_complex_free(P[i]);
        gsl_vector_complex_free(K[i]);

        gsl_vector_complex_free(P_timelike[i]);

        gsl_vector_complex_free(p_i[i]);
        gsl_vector_complex_free(p_f[i]);
        gsl_vector_complex_free(k_i[i]);
        gsl_vector_complex_free(k_f[i]);

        gsl_vector_complex_free(p_i_timelike[i]);
        gsl_vector_complex_free(p_f_timelike[i]);
        gsl_vector_complex_free(k_i_timelike[i]);
        gsl_vector_complex_free(k_f_timelike[i]);

        gsl_vector_complex_free(p_ext[i]);
        gsl_vector_complex_free(k_ext[i]);
        gsl_vector_complex_free(q_ext[i]);

        gsl_vector_complex_free(q_ext_timelike[i]);
    }

    delete []Q;
    delete []P;
    delete []K;

    delete []P_timelike;

    delete []p_i;
    delete []p_f;
    delete []k_i;
    delete []k_f;

    delete []p_i_timelike;
    delete []p_f_timelike;
    delete []k_i_timelike;
    delete []k_f_timelike;

    delete []p_ext;
    delete []k_ext;
    delete []q_ext;

    delete []q_ext_timelike;
}

void ExternalImpulseGrid::calc_q_ext_timelike(gsl_vector_complex* q_ext_timelike, double X, gsl_complex nucleon_mass)
{
    assert(X > 0);

    gsl_vector_complex_set_zero(q_ext_timelike);
    gsl_vector_complex_set(q_ext_timelike, 3, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul_imag(nucleon_mass, 2.0 * sqrt(1 + X));

    gsl_vector_complex_scale(q_ext_timelike, pref);

    assert(GSL_IMAG(pref) > 0);
    assert(GSL_REAL(pref) == 0);
}

void ExternalImpulseGrid::calc_q_ext(gsl_vector_complex* q_ext, double X, gsl_complex nucleon_mass, double a)
{
    assert(X > 0);

    gsl_vector_complex_set_zero(q_ext);
    gsl_vector_complex_set(q_ext, 3, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul_real(nucleon_mass, 2.0 * sqrt(1.0 + X));
    pref = gsl_complex_mul(pref, gsl_complex_sqrt_real(a));

    gsl_vector_complex_scale(q_ext, pref);

    assert(GSL_IMAG(pref) == 0);
    assert(GSL_REAL(pref) > 0);
}

void ExternalImpulseGrid::calc_p_ext(gsl_vector_complex* p_ext, double X, gsl_complex nucleon_mass)
{
    assert(X > 0);

    gsl_vector_complex_set_zero(p_ext);
    gsl_vector_complex_set(p_ext, 2, gsl_complex_rect(1, 0));

    gsl_complex pref = gsl_complex_mul_real(nucleon_mass, sqrt(X));

    gsl_vector_complex_scale(p_ext, pref);

    assert(GSL_IMAG(pref) == 0);
    assert(GSL_REAL(pref) > 0);
}

void ExternalImpulseGrid::calc_k_ext(gsl_vector_complex* k_ext, double X, gsl_complex nucleon_mass, double z)
{
    assert(X > 0);

    gsl_vector_complex_set_zero(k_ext);
    gsl_vector_complex_set(k_ext, 1, gsl_complex_sqrt_real(1 - gsl_pow_2(z)));
    gsl_vector_complex_set(k_ext, 2, gsl_complex_rect(z, 0));

    gsl_complex pref = gsl_complex_mul_real(nucleon_mass, sqrt(X));

    gsl_vector_complex_scale(k_ext, pref);

    assert(GSL_IMAG(pref) == 0);
    assert(GSL_REAL(pref) > 0);
}

void ExternalImpulseGrid::calc_Q(gsl_vector_complex* Q, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(Q, p_ext);
    gsl_vector_complex_sub(Q, k_ext);
}

void ExternalImpulseGrid::calc_P(gsl_vector_complex* P, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(P, q_ext);
    gsl_vector_complex_add(P, p_ext);
    gsl_vector_complex_add(P, k_ext);

    gsl_vector_complex_scale(P, gsl_complex_rect(0.5, 0));
}

void ExternalImpulseGrid::calc_K(gsl_vector_complex* K, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(K, q_ext);
    gsl_vector_complex_sub(K, k_ext);
    gsl_vector_complex_sub(K, p_ext);

    gsl_vector_complex_scale(K, gsl_complex_rect(0.5, 0));
}

void ExternalImpulseGrid::calc_p_i(gsl_vector_complex* p_i, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(p_i, q_ext);
    gsl_vector_complex_scale(p_i, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_add(p_i, k_ext);
}

void ExternalImpulseGrid::calc_p_f(gsl_vector_complex* p_f, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(p_f, q_ext);
    gsl_vector_complex_scale(p_f, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_add(p_f, p_ext);
}

void ExternalImpulseGrid::calc_k_i(gsl_vector_complex* k_i, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(k_i, q_ext);
    gsl_vector_complex_scale(k_i, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_sub(k_i, k_ext);
}

void ExternalImpulseGrid::calc_k_f(gsl_vector_complex* k_f, const gsl_vector_complex* k_ext, const gsl_vector_complex* p_ext, const gsl_vector_complex* q_ext)
{
    gsl_vector_complex_memcpy(k_f, q_ext);
    gsl_vector_complex_scale(k_f, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_sub(k_f, p_ext);
}

gsl_vector_complex* ExternalImpulseGrid::get_Q(int idx)
{
    return Q[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_P(int idx)
{
    return P[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_P_timelike(int idx)
{
    return P_timelike[idx];
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

gsl_vector_complex* ExternalImpulseGrid::get_p_i_timelike(int idx)
{
    return p_i_timelike[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_p_f_timelike(int idx)
{
    return p_f_timelike[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_k_i_timelike(int idx)
{
    return k_i_timelike[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_k_f_timelike(int idx)
{
    return k_f_timelike[idx];
}

double ExternalImpulseGrid::calc_tau(int XIdx, int zIdx)
{
    int gridIdx = getGridIdx(XIdx, zIdx);

    gsl_complex Q2;
    gsl_blas_zdotu(Q[gridIdx], Q[gridIdx], &Q2);

    assert(GSL_IMAG(Q2) == 0);
    assert(GSL_REAL(Q2) > 0);

    gsl_complex M2 = gsl_complex_pow_real(nucleon_mass, 2);
    assert(GSL_IMAG(M2) == 0);

    double tau = GSL_REAL(Q2)/(4.0 * GSL_REAL(M2));
    return tau;
}


