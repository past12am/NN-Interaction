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
#include "momentumloops/MomentumLoop.hpp"

class ScatteringProcess
{
    private:
        gsl_vector_complex* k;
        std::mutex k_mutex;

        gsl_matrix_complex* inverseKMatrix;

    protected:
        int threadIdx;

        gsl_complex nucleon_mass;

        gsl_complex* scattering_amplitude_basis_projected;  // h_i
        gsl_complex* form_factors;                          // f_i

        Tensor4<4, 4, 4, 4>* scattering_amplitude;

        ExternalImpulseGrid externalImpulseGrid;
        TensorBasis tensorBasis;

        MomentumLoop* momentumLoop;

    public:
        ScatteringProcess(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, gsl_complex nucleon_mass, int threadIdx);
        virtual ~ScatteringProcess();

        void performScatteringCalculation(double k2_cutoff);
        void buildScatteringMatrix();

        void calculateFormFactors(int XIdx, int ZIdx, gsl_complex M, gsl_vector_complex* f);
        void build_h_vector(int externalImpulseIdx, gsl_vector_complex* h);

        TensorBasis* getTensorBasis();

        int calcScatteringAmpIdx(int basisElemIdx, int externalImpulseIdx);

        void store_scattering_amplitude(int basisElemIdx, std::ofstream& data_file);
        double calcSquaredNormOfScatteringMatrix(int externalImpulseIdx);


        gsl_complex integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double k2, double z, double y, double phi);

        virtual void integrate(double k2_cutoff);


        virtual void integralKernel(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P,
                                    gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                    gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                    Tensor4<4, 4, 4, 4>* integralKernelTensor) = 0;

        virtual gsl_complex integrate_process(int basisElemIdx, int externalImpulseIdx, double k2_cutoff) = 0;

};

#endif //NNINTERACTION_SCATTERINGPROCESS_HPP
