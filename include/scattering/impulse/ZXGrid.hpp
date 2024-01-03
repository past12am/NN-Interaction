//
// Created by past12am on 10/7/23.
//

#ifndef NNINTERACTION_ZXGRID_HPP
#define NNINTERACTION_ZXGRID_HPP


#include <gsl/gsl_complex.h>

class ZXGrid
{
    protected:
        int lenX;
        int lenZ;

        double XCutoffLower;
        double XCutoffUpper;

        double ZCutoffLower;
        double ZCutoffUpper;

        double* X;
        double* Z;


    public:
        ZXGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double ZCutoffLower, double ZCutoffUpper);
        virtual ~ZXGrid();

        double calcZAt(int ZIdx);
        double calcXAt(int XIdx);

        double getZAt(int ZIdx);
        double getXAt(int XIdx);

        int getGridIdx(int XIdx, int ZIdx);

        int getLenX() const;
        int getLenZ() const;
        int getLength() const;
};


#endif //NNINTERACTION_ZXGRID_HPP
