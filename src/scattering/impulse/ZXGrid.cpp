//
// Created by past12am on 10/7/23.
//

#include <cassert>
#include <gsl/gsl_complex_math.h>
#include "../../../include/scattering/impulse/ZXGrid.hpp"


int ZXGrid::getGridIdx(int XIdx, int zIdx)
{
    return XIdx * lenZ + zIdx;
}

int ZXGrid::getLength() const
{
    return lenX * lenZ;
}

double ZXGrid::calcZAt(int zIdx)
{
    return zCutoffLower + (zCutoffUpper - zCutoffLower) * ((double) zIdx)/((double) (lenZ - 1));
}

double ZXGrid::calcXAt(int XIdx)
{
    return XCutoffLower + (XCutoffUpper - XCutoffLower) * ((double) XIdx) / ((double) (lenX - 1));
}

int ZXGrid::getLenX() const
{
    return lenX;
}

int ZXGrid::getLenZ() const
{
    return lenZ;
}

ZXGrid::ZXGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double zCutoffLower,
               double zCutoffUpper) : lenX(lenX), lenZ(lenZ),
                                          XCutoffLower(XCutoffLower),
                                          XCutoffUpper(XCutoffUpper),
                                          zCutoffLower(zCutoffLower),
                                          zCutoffUpper(zCutoffUpper)
{
    X = new double[lenX];
    z = new double[lenZ];

    for(int XIdx = 0; XIdx < lenX; XIdx++)
    {
        X[XIdx] = calcXAt(XIdx);

        for (int zIdx = 0; zIdx < lenZ; zIdx++)
        {
            z[zIdx] = calcZAt(zIdx);
        }
    }

    assert(X[0] == XCutoffLower);
    //assert(X[lenX - 1] == XCutoffUpper);
    assert(z[0] == zCutoffLower);
    assert(z[lenZ - 1] == zCutoffUpper);
}

ZXGrid::~ZXGrid()
{
    delete[] X;
    delete[] z;
}

double ZXGrid::getZAt(int zIdx)
{
    return z[zIdx];
}

double ZXGrid::getXAt(int XIdx)
{
    return X[XIdx];
}
