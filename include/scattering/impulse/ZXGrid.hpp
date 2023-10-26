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

        double zCutoffLower;
        double zCutoffUpper;

        double* X;
        double* z;


    public:
        ZXGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower, double zCutoffUpper);

        virtual ~ZXGrid();

        double calcZAt(int zIdx);
        double calcXAt(int XIdx);

        double getZAt(int zIdx);
        double getXAt(int XIdx);

        int getGridIdx(int XIdx, int zIdx);

        int getLenX() const;
        int getLenZ() const;
        int getLength() const;
};


#endif //NNINTERACTION_ZXGRID_HPP
