//
// Created by past12am on 8/3/23.
//

#ifndef NNINTERACTION_SCATTERINGPROCESS_HPP
#define NNINTERACTION_SCATTERINGPROCESS_HPP

#include <complex>
#include <gsl/gsl_vector.h>
#include <mutex>
#include "impulse/ExternalImpulseGrid.hpp"
#include "basis/TensorBasis.hpp"
#include "../numerics/Integratable.hpp"

class ScatteringProcess
{
    private:
        gsl_vector_complex* l;
        std::mutex l_mutex;

        gsl_matrix_complex* inverseKMatrix;

    protected:
        int threadIdx;

        gsl_complex nucleon_mass;

        gsl_complex* scattering_amplitude_basis_projected;  // h_i
        gsl_complex* form_factors;                          // f_i

        Tensor22<4, 4, 4, 4>* scattering_matrix;

        ExternalImpulseGrid externalImpulseGrid;
        TensorBasis tensorBasis;

        void calc_l(double l2, double z, double y, double phi, gsl_vector_complex* l);

    public:
        ScatteringProcess(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, double a, int threadIdx);
        virtual ~ScatteringProcess();


        TensorBasis* getTensorBasis();

        void store_scattering_amplitude(int basisElemIdx, double a, std::ofstream& data_file);

        void performScatteringCalculation(double l2_cutoff);

        void buildScatteringMatrix();

        int calcScatteringAmpIdx(int basisElemIdx, int externalImpulseIdx);

        [[deprecated("Use Tensor Basis stored inverse K element")]]
        gsl_matrix_complex* buildInverseK(double tau, double z, gsl_complex M);

        void build_h_vector(int externalImpulseIdx, gsl_vector_complex* h);

        void calculateFormFactors(int XIdx, int zIdx, gsl_complex M, gsl_vector_complex* f);

        double calcSquaredNormOfScatteringMatrix(int externalImpulseIdx);

        gsl_complex integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double l2, double z, double y, double phi);


        virtual void integralKernel(gsl_vector_complex* l, gsl_vector_complex* Q, gsl_vector_complex* K, gsl_vector_complex* P,
                                    gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                    gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                    Tensor22<4, 4, 4, 4>* integralKernelTensor) = 0;

        virtual void integrate(double l2_cutoff) = 0;
};

#endif //NNINTERACTION_SCATTERINGPROCESS_HPP
