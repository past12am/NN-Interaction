//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_SCATTERINGPROCESS_HPP
#define NNINTERACTION_SCATTERINGPROCESS_HPP

#include <complex>
#include <gsl/gsl_vector.h>
#include "impulse/ExternalImpulseGrid.hpp"
#include "basis/TensorBasis.hpp"
#include "../numerics/Integratable.hpp"

class ScatteringProcess
{
    protected:
        ExternalImpulseGrid externalImpulseGrid;
        TensorBasis tensorBasis;

        void calc_l(double l2, double z, double y, double phi, gsl_vector_complex* l);

    public:
        virtual void integralKernel(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* K, gsl_vector_complex* P,
                                    gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                    gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                    Tensor4<4, 4, 4, 4>* integralKernelTensor) = 0;

        ScatteringProcess(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, gsl_complex M_nucleon);
};

#endif //NNINTERACTION_SCATTERINGPROCESS_HPP
