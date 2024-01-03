//
// Created by past12am on 8/25/23.
//

#include "../../../include/scattering/impulse/LoopImpulseGrid.hpp"

#include "math.h"
#include "iostream"

LoopImpulseGrid::LoopImpulseGrid(int length, double k2_grid_start, double k2_grid_center, double k2_grid_end) : length(length), k2_grid_start(k2_grid_start), k2_grid_center(k2_grid_center), k2_grid_end(k2_grid_end)
{
    k2 = new double[length];

    populateGridLogarithmicDistribution();
}

LoopImpulseGrid::~LoopImpulseGrid()
{
    delete k2;
}

void LoopImpulseGrid::populateGridLogarithmicDistribution()
{
    double p2_grid_exp_center = log10(k2_grid_center);
    double p2_grid_exp_start = log10(k2_grid_start);
    double p2_grid_exp_end = log10(k2_grid_end);

    double dist_lower = (p2_grid_exp_start - p2_grid_exp_center) / ((length + 1)/2 - 1);
    double dist_upper = (p2_grid_exp_end - p2_grid_exp_center) / ((length + 1)/2);

    for (int i = 0; i < length/2; i++)
    {
        double impulseLowerHalf = pow(10.0, p2_grid_exp_center + (i) * dist_lower);
        double impulseUpperHalf = pow(10.0, p2_grid_exp_center + (i + 1) * dist_upper);

        k2[(length) / 2 - 1 - i] = impulseLowerHalf;
        k2[(length) / 2 + i] = impulseUpperHalf;
    }

    std::cout << "p2_grid: ";
    for (int i = 0; i < length; i++)
    {
        std::cout << k2[i] << " - ";
    }
    std::cout << std::endl << std::endl;
}

int LoopImpulseGrid::getLength() const
{
    return length;
}
