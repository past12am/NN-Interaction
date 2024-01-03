//
// Created by past12am on 8/25/23.
//

#ifndef NNINTERACTION_LOOPIMPULSEGRID_HPP
#define NNINTERACTION_LOOPIMPULSEGRID_HPP

#include <complex>
#include <gsl/gsl_vector.h>

class LoopImpulseGrid
{
    private:
        int length;

        double k2_grid_start;
        double k2_grid_center;
        double k2_grid_end;

        double* k2;

        void populateGridLogarithmicDistribution();

    public:
        LoopImpulseGrid(int length, double k2_grid_start, double k2_grid_center, double k2_grid_end);
        virtual ~LoopImpulseGrid();

        int getLength() const;
};


#endif //NNINTERACTION_LOOPIMPULSEGRID_HPP
