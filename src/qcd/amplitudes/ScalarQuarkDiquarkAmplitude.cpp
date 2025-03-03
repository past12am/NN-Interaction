//
// Created by past12am on 8/2/23.
//

#include <complex>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

#include "../../../include/utils/dirac/DiracStructuresHelper.hpp"
#include "../../../include/qcd/amplitudes/ScalarQuarkDiquarkAmplitude.hpp"
#include "../../../include/operators/ChargeConjugation.hpp"


QuarkDiquarkAmplitudeReader* ScalarQuarkDiquarkAmplitude::fit_reader = nullptr;


ScalarQuarkDiquarkAmplitude::ScalarQuarkDiquarkAmplitude()
{
    ScalarQuarkDiquarkAmplitude::fit_reader = QuarkDiquarkAmplitudeReader::getInstance();

    posEnergyProj = gsl_matrix_complex_alloc(4, 4);
    tmpTensor = gsl_matrix_complex_alloc(4, 4);
    NLOTensor = gsl_matrix_complex_alloc(4, 4);

    q = gsl_vector_complex_alloc(4);

    p_copy = gsl_vector_complex_alloc(4);
    P_copy = gsl_vector_complex_alloc(4);
}

ScalarQuarkDiquarkAmplitude::~ScalarQuarkDiquarkAmplitude()
{
    gsl_vector_complex_free(P_copy);
    gsl_vector_complex_free(p_copy);

    gsl_vector_complex_free(q);

    gsl_matrix_complex_free(NLOTensor);
    gsl_matrix_complex_free(tmpTensor);
    gsl_matrix_complex_free(posEnergyProj);
}

void ScalarQuarkDiquarkAmplitude::Gamma(gsl_vector_complex* p, gsl_vector_complex* P, bool chargeConj, int threadIdx, gsl_matrix_complex* quarkDiquarkAmp)
{
    // Charge Conjugation
    //  ChargeConj(Phi(p, P)) = C Phi(-p, -P)^T C^T

    gsl_vector_complex_memcpy(p_copy, p);
    gsl_vector_complex_memcpy(P_copy, P);

    if(chargeConj)
    {
        gsl_vector_complex_scale(p_copy, gsl_complex_rect(-1.0, 0));
        gsl_vector_complex_scale(P_copy, gsl_complex_rect(-1.0, 0));
    }

    // Get quantities needed for all tensor orders
    gsl_complex p2;
    gsl_blas_zdotu(p_copy, p_copy, &p2);

    gsl_complex P2;
    gsl_blas_zdotu(P_copy, P_copy, &P2);


    gsl_complex compl_z;
    gsl_blas_zdotu(p_copy, P_copy, &compl_z);
    if(GSL_IMAG(compl_z) - 1 > 1E-15)
    {
        throw std::out_of_range("Encountered complex angle for quark-diquark amplitude momenta");
    }
    double z = GSL_REAL(compl_z) / (sqrt(gsl_complex_abs(p2) * gsl_complex_abs(P2)));


    Projectors::posEnergyProjector(P_copy, posEnergyProj);


    // 0: Leading Tensor    ( = unity)
    // quarkDiquarkAmp = f(p2, z, 0) * posEnergyProj(P)
    gsl_matrix_complex_memcpy(quarkDiquarkAmp, posEnergyProj);

    gsl_complex f_k_0 = fit_reader->f_k(p2, z, 0);
    gsl_matrix_complex_scale(quarkDiquarkAmp, f_k_0);



    //  TODO higher order tensor seems to be too noisy in the result
    /*
    // 1: Higher order Tensor ( = slash(i * normalized(TransverseProj_P @ p)))
    // build slash(q)   q = normalized(TransverseProj_P(p))
    // tmpTensor = TransverseProj_P
    Projectors::transverseProjector(P, tmpTensor);

    // q = TransverseProj_P @ p
    gsl_vector_complex_set_zero(q);
    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, tmpTensor, p, GSL_COMPLEX_ZERO, q);

    // q = i normalize(q) = i * 1/sqrt(q2) TransverseProj_P @ p
    gsl_complex q2;
    gsl_blas_zdotu(q, q, &q2);
    //gsl_vector_complex_scale(q, gsl_complex_mul_imag(gsl_complex_inverse(gsl_complex_sqrt(q2)), 1));
    gsl_vector_complex_scale(q, gsl_complex_rect(1.0/sqrt(sqrt(gsl_complex_abs(q2))), 0));

    // tmpTensor = slash(q)
    DiracStructuresHelper::diracStructures.slash(q, tmpTensor);


    // scale by f() and multiply with pos energy projector
    // NLOTensor = tmpTensor @ posEnergyProj
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, tmpTensor, posEnergyProj, GSL_COMPLEX_ZERO, NLOTensor);
    gsl_matrix_complex_scale(NLOTensor, fit_reader->f_k(p2, z, 1));



    // Add tensor contibutions together
    gsl_matrix_complex_add(quarkDiquarkAmp, NLOTensor);
    */

    if(chargeConj)
    {
        ChargeConjugation::chargeConj(quarkDiquarkAmp, threadIdx);
    }
}

gsl_complex ScalarQuarkDiquarkAmplitude::f(gsl_complex p2)
{
    // f = (c1 + c2*p^2) * e^(-c3*p^2)
    gsl_complex term1 = gsl_complex_add(c1, gsl_complex_mul(c2, p2));
    gsl_complex exponential_param = gsl_complex_mul(gsl_complex_negative(c3), p2);
    gsl_complex term2 = gsl_complex_exp(exponential_param);

    return gsl_complex_mul(term1, term2);
}
