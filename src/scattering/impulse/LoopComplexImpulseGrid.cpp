//
// Created by past12am on 8/25/23.
//

#include "../../../include/scattering/impulse/LoopComplexImpulseGrid.hpp"

#include "complex"
#include "math.h"
#include "gsl/gsl_complex_math.h"
#include <iostream>

LoopComplexImpulseGrid::LoopComplexImpulseGrid(int lenReal, int lenImag, gsl_complex p2_grid_start, gsl_complex p2_grid_center, gsl_complex p2_grid_end) : lenReal(lenReal), lenImag(lenImag),
                                                                                                                                                           p2_grid_start(p2_grid_start), p2_grid_center(p2_grid_center), p2_grid_end(p2_grid_end)
{
    int len = getLength();
    l2 = new gsl_complex[len];

    populateGridLogarithmicDistribution();
}

LoopComplexImpulseGrid::~LoopComplexImpulseGrid()
{
    delete l2;
}

int LoopComplexImpulseGrid::getLenReal() const
{
    return lenReal;
}

int LoopComplexImpulseGrid::getLenImag() const
{
    return lenImag;
}

int LoopComplexImpulseGrid::getLength() const
{
    return lenReal * lenImag;
}

int LoopComplexImpulseGrid::getGridIndex(int realIdx, int imagIdx)
{
    return realIdx * lenImag + imagIdx;
}

void LoopComplexImpulseGrid::populateGridLogarithmicDistribution()
{
    double p2_grid_real_exp_center = log10(GSL_REAL(p2_grid_center));
    double p2_grid_real_exp_start = log10(GSL_REAL(p2_grid_start));
    double p2_grid_real_exp_end = log10(GSL_REAL(p2_grid_end));

    double dist_upper_real = (p2_grid_real_exp_end - p2_grid_real_exp_center) / ((lenReal + 1)/2 - 1);
    double dist_lower_real = (p2_grid_real_exp_start - p2_grid_real_exp_center) / ((lenReal + 1)/2 - 1);

    for (int realIdx = 0; realIdx < lenReal/2; realIdx++)
    {
        double realPartLowerHalf = pow(10.0, p2_grid_real_exp_center + (realIdx) * dist_lower_real);
        double realPartUpperHalf = pow(10.0, p2_grid_real_exp_center + (realIdx) * dist_upper_real);

        int iLower =  (lenReal)/2 - 1 - realIdx;
        int iUpper =  (lenReal)/2 + realIdx;

        populateImaginarySubGridLogarithmicDistribution(iLower, realPartLowerHalf);
        populateImaginarySubGridLogarithmicDistribution(iUpper, realPartUpperHalf);
    }

    /*
    std::cout << "p2_grid: ";
    for (int realIdx = 0; realIdx < lenReal; realIdx++)
    {
        for (int imagIdx = 0; imagIdx < lenImag; imagIdx++)
        {
            int i = getGridIndex(realIdx, imagIdx);
            std::cout << GSL_REAL(l2[i]) << " " << GSL_IMAG(l2[i]) << " - ";
        }
    }
    std::cout << std::endl << std::endl;
    */
}

void LoopComplexImpulseGrid::populateImaginarySubGridLogarithmicDistribution(int realChunkIdx, double realPart)
{
    double p2_grid_imag_exp_center = log10(GSL_IMAG(p2_grid_center));
    double p2_grid_imag_exp_start = log10(GSL_IMAG(p2_grid_start));
    double p2_grid_imag_exp_end = log10(GSL_IMAG(p2_grid_end));

    double dist_lower_imag = (p2_grid_imag_exp_start - p2_grid_imag_exp_center) / ((lenImag + 1)/2 - 1);
    double dist_upper_imag = (p2_grid_imag_exp_end - p2_grid_imag_exp_center) / ((lenImag + 1)/2);

    for (int imagIdx = 0; imagIdx < lenImag/2; imagIdx++)
    {
        double imagPartLowerHalf = pow(10.0, p2_grid_imag_exp_center + (imagIdx) * dist_lower_imag);
        double imagPartUpperHalf = pow(10.0, p2_grid_imag_exp_center + (imagIdx + 1) * dist_upper_imag);

        int iLower = getGridIndex(realChunkIdx, (lenImag)/2 - 1 - imagIdx);
        int iUpper = getGridIndex(realChunkIdx, (lenImag)/2 + imagIdx);

        l2[iLower] = gsl_complex_rect(realPart, imagPartLowerHalf);
        l2[iUpper] = gsl_complex_rect(realPart, imagPartUpperHalf);
    }
}
