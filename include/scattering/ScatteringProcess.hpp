//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_SCATTERINGPROCESS_HPP
#define NNINTERACTION_SCATTERINGPROCESS_HPP

#include <complex>
#include <gsl/gsl_vector.h>
#include "../helpers/ExternalImpulseGrid.hpp"
#include "basis/TensorBasis.hpp"

class ScatteringProcess
{
    protected:
        ExternalImpulseGrid externalImpulseGrid;
        TensorBasis tensorBasis;

    public:
        virtual void integralKernel(gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                    gsl_vector_complex* p_r, gsl_vector_complex* p_rp,
                                    gsl_vector_complex* p_q, gsl_vector_complex* p_d,
                                    gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                    gsl_vector_complex* k_r, gsl_vector_complex* k_rp,
                                    gsl_vector_complex* k_q, gsl_vector_complex* k_d,
                                    gsl_complex mu2) = 0;

        ScatteringProcess(int lenTau, int lenZ, double tauCutoff, double m, double M);
};

#endif //NNINTERACTION_SCATTERINGPROCESS_HPP
