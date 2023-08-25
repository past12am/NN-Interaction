//
// Created by past12am on 8/25/23.
//

#ifndef NNINTERACTION_LOOPCOMPLEXIMPULSEGRID_HPP
#define NNINTERACTION_LOOPCOMPLEXIMPULSEGRID_HPP

#include <complex>
#include <gsl/gsl_matrix.h>

class LoopComplexImpulseGrid
{
    private:
        int lenReal;
        int lenImag;

        gsl_complex p2_grid_start;
        gsl_complex p2_grid_center;
        gsl_complex p2_grid_end;

        gsl_complex* l2;

        int getGridIndex(int realIdx, int imagIdx);

        void populateGridLogarithmicDistribution();
        void populateImaginarySubGridLogarithmicDistribution(int realChunkIdx, double realPart);

    public:
        LoopComplexImpulseGrid(int lenReal, int lenComplex, gsl_complex p2_grid_start, gsl_complex p2_grid_center, gsl_complex p2_grid_end);
        virtual ~LoopComplexImpulseGrid();

        int getLenReal() const;
        int getLenImag() const;
        int getLength() const;
};


#endif //NNINTERACTION_LOOPCOMPLEXIMPULSEGRID_HPP
