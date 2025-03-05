//
// Created by past12am on 8/18/23.
//

#include <cassert>
#include <gsl/gsl_blas.h>
#include "../../../include/scattering/impulse/ExternalImpulseGrid.hpp"

#include "complex"
#include "../../../include/Definitions.h"
#include "gsl/gsl_complex_math.h"
#include "gsl/gsl_math.h"

ExternalImpulseGrid::ExternalImpulseGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper) :
        ZXGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, zCutoffLower, zCutoffUpper)
{
    int len = getLength();

    l_ext = new gsl_vector_complex*[len];
    r_ext = new gsl_vector_complex*[len];
    P_ext = new gsl_vector_complex*[len];

    p_i = new gsl_vector_complex*[len];
    p_f = new gsl_vector_complex*[len];
    k_i = new gsl_vector_complex*[len];
    k_f = new gsl_vector_complex*[len];

    for(int XIdx = 0; XIdx < lenX; XIdx++)
    {
        for(int ZIdx = 0; ZIdx < lenZ; ZIdx++)
        {
            int i = getGridIdx(XIdx, ZIdx);

            l_ext[i] = gsl_vector_complex_alloc(4);
            r_ext[i] = gsl_vector_complex_alloc(4);
            P_ext[i] = gsl_vector_complex_alloc(4);
            calc_l_ext(l_ext[i], getXAt(XIdx), getZAt(ZIdx));
            calc_r_ext(r_ext[i], getXAt(XIdx), getZAt(ZIdx));
            calc_P_ext(P_ext[i], getXAt(XIdx), getZAt(ZIdx));

            p_i[i] = gsl_vector_complex_alloc(4);
            p_f[i] = gsl_vector_complex_alloc(4);
            k_i[i] = gsl_vector_complex_alloc(4);
            k_f[i] = gsl_vector_complex_alloc(4);
            calc_p_i(p_i[i], l_ext[i], r_ext[i], P_ext[i]);
            calc_p_f(p_f[i], l_ext[i], r_ext[i], P_ext[i]);
            calc_k_i(k_i[i], l_ext[i], r_ext[i], P_ext[i]);
            calc_k_f(k_f[i], l_ext[i], r_ext[i], P_ext[i]);
        }
    }
}

ExternalImpulseGrid::~ExternalImpulseGrid()
{
    for(int i = 0; i < getLength(); i++)
    {
        gsl_vector_complex_free(p_i[i]);
        gsl_vector_complex_free(p_f[i]);
        gsl_vector_complex_free(k_i[i]);
        gsl_vector_complex_free(k_f[i]);

        gsl_vector_complex_free(l_ext[i]);
        gsl_vector_complex_free(r_ext[i]);
        gsl_vector_complex_free(P_ext[i]);
    }

    delete []p_i;
    delete []p_f;
    delete []k_i;
    delete []k_f;

    delete []l_ext;
    delete []r_ext;
    delete []P_ext;
}

void ExternalImpulseGrid::calc_l_ext(gsl_vector_complex* l_ext, double X, double Z)
{
    assert(X > 0);
    assert(X < 1);

    gsl_vector_complex_set_zero(l_ext);
    gsl_vector_complex_set(l_ext, 2, GSL_COMPLEX_ONE);

    gsl_complex pref = gsl_complex_rect(M_nucleon * sqrt(X), 0); //gsl_complex_mul_real(nucleon_mass, sqrt(X));
    gsl_vector_complex_scale(l_ext, pref);

    assert(GSL_REAL(pref) > 0);
    assert(GSL_IMAG(pref) == 0);
}

void ExternalImpulseGrid::calc_r_ext(gsl_vector_complex* r_ext, double X, double Z)
{
    assert(X > 0);
    assert(X < 1);

    gsl_vector_complex_set_zero(r_ext);
    gsl_vector_complex_set(r_ext, 1, gsl_complex_sqrt_real(1.0 - gsl_pow_2(Z)));
    gsl_vector_complex_set(r_ext, 2, gsl_complex_rect(Z, 0));

    gsl_complex pref = gsl_complex_rect(M_nucleon * sqrt(X), 0); //gsl_complex_mul_real(nucleon_mass, sqrt(X));
    gsl_vector_complex_scale(r_ext, pref);

    assert(GSL_REAL(pref) > 0);
    assert(GSL_IMAG(pref) == 0);
}

void ExternalImpulseGrid::calc_P_ext(gsl_vector_complex* P_ext, double X, double Z)
{
    assert(X > 0);
    assert(X < 1);

    gsl_vector_complex_set_zero(P_ext);
    gsl_vector_complex_set(P_ext, 3, GSL_COMPLEX_ONE);

    gsl_complex pref = gsl_complex_rect(0, 2.0 * M_nucleon * sqrt(1.0 + X)); //gsl_complex_mul(nucleon_mass, gsl_complex_rect(0, 2.0 * sqrt(1.0 + X)));
    gsl_vector_complex_scale(P_ext, pref);

    assert(GSL_IMAG(pref) > 0);
    assert(GSL_REAL(pref) == 0);
}


void ExternalImpulseGrid::calc_p_i(gsl_vector_complex* p_i, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext)
{
    // p_i = P/2 + r
    gsl_vector_complex_memcpy(p_i, P_ext);
    gsl_vector_complex_scale(p_i, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_add(p_i, r_ext);
}

void ExternalImpulseGrid::calc_p_f(gsl_vector_complex* p_f, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext)
{
    // p_f = P/2 + l
    gsl_vector_complex_memcpy(p_f, P_ext);
    gsl_vector_complex_scale(p_f, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_add(p_f, l_ext);
}

void ExternalImpulseGrid::calc_k_i(gsl_vector_complex* k_i, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext)
{
    // k_i = P/2 - r
    gsl_vector_complex_memcpy(k_i, P_ext);
    gsl_vector_complex_scale(k_i, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_sub(k_i, r_ext);
}

void ExternalImpulseGrid::calc_k_f(gsl_vector_complex* k_f, const gsl_vector_complex* l_ext, const gsl_vector_complex* r_ext, const gsl_vector_complex* P_ext)
{
    // k_f = P/2 - l
    gsl_vector_complex_memcpy(k_f, P_ext);
    gsl_vector_complex_scale(k_f, gsl_complex_rect(0.5, 0));
    gsl_vector_complex_sub(k_f, l_ext);
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

gsl_vector_complex* ExternalImpulseGrid::get_l_ext(int idx)
{
    return l_ext[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_r_ext(int idx)
{
    return r_ext[idx];
}

gsl_vector_complex* ExternalImpulseGrid::get_P_ext(int idx)
{
    return P_ext[idx];
}


