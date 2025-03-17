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

        gsl_complex* scattering_amplitude_basis_projected;  // h_i
        gsl_complex* form_factors;                          // f_i

        Tensor4<4, 4, 4, 4>* scattering_amplitude;

        int len_contour_def_epsilon;
        double* contour_def_epsilon;

        ExternalImpulseGrid externalImpulseGrid;
        TensorBasis tensorBasis;

        MomentumLoop* momentumLoop;

    public:
        ScatteringProcess(int lenX, int lenZ, int lenContourDef, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper, double contourEpsLower, double contourEpsUpper, int threadIdx);
        virtual ~ScatteringProcess();

        void performScatteringCalculation();
        void buildScatteringMatrix();

        void calculateFormFactors(int contour_def_idx, int XIdx, int ZIdx, gsl_vector_complex* f);
        void build_h_vector(int contour_def_idx, int externalImpulseIdx, gsl_vector_complex* h);

        TensorBasis* getTensorBasis();

        int calcScatteringAmpIdx(int basisElemIdx, int contour_def_idx, int externalImpulseIdx);

        void store_scattering_amplitude(int basisElemIdx, std::ofstream& data_file);
        double calcSquaredNormOfScatteringMatrix(int contour_def_idx, int externalImpulseIdx);


        gsl_complex integralKernelWrapper(int externalImpulseIdx, int basisElemIdx, int threadIdx, double k2, double z, double y, double phi);
        gsl_complex deformedIntegralKernelWrapper(int externalImpulseIdx, int contourDefEpsIdx, int basisElemIdx, int threadIdx, gsl_complex x_4, double absx, double y, double phi);

        virtual void integrate();


        virtual void integralKernel(gsl_vector_complex* k, gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P,
                                    gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                    gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                    Tensor4<4, 4, 4, 4>* integralKernelTensor) = 0;
        virtual void deformedIntegralKernel(gsl_vector_complex* k, gsl_complex x_4, double absx, double y, double phi, double epsilon,
                                            double X, double Z,
                                            gsl_vector_complex* l, gsl_vector_complex* r, gsl_vector_complex* P,
                                            gsl_vector_complex* p_f, gsl_vector_complex* p_i,
                                            gsl_vector_complex* k_f, gsl_vector_complex* k_i,
                                            Tensor4<4, 4, 4, 4>* integralKernelTensor) = 0;

        // TODO proceed from here with epsilon implementation
        virtual gsl_complex integrate_process(int basisElemIdx, int contourDefEpsIdx, int externalImpulseIdx) = 0;

};

#endif //NNINTERACTION_SCATTERINGPROCESS_HPP
