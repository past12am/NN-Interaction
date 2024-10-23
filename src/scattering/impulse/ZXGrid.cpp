//
// Created by past12am on 10/7/23.
//

#include <cassert>
#include "../../../include/scattering/impulse/ZXGrid.hpp"


int ZXGrid::getGridIdx(int XIdx, int ZIdx)
{
    return XIdx * lenZ + ZIdx;
}

int ZXGrid::getLength() const
{
    return lenX * lenZ;
}

double ZXGrid::calcZAt(int ZIdx)
{
    return ZCutoffLower + (ZCutoffUpper - ZCutoffLower) * ((double) ZIdx) / ((double) (lenZ - 1));
}

double ZXGrid::calcXAt(int XIdx)
{
    return lenX == 1 ? XCutoffLower : XCutoffLower + (XCutoffUpper - XCutoffLower) * ((double) XIdx) / ((double) (lenX - 1));
}

int ZXGrid::getLenX() const
{
    return lenX;
}

int ZXGrid::getLenZ() const
{
    return lenZ;
}

ZXGrid::ZXGrid(int lenX, int lenZ, double XCutoffLower, double XCutoffUpper, double ZCutoffLower,
               double ZCutoffUpper) : lenX(lenX), lenZ(lenZ),
                                      XCutoffLower(XCutoffLower),
                                      XCutoffUpper(XCutoffUpper),
                                      ZCutoffLower(ZCutoffLower),
                                      ZCutoffUpper(ZCutoffUpper)
{
    X = new double[lenX];
    Z = new double[lenZ];

    for(int XIdx = 0; XIdx < lenX; XIdx++)
    {
        X[XIdx] = calcXAt(XIdx);

        for (int zIdx = 0; zIdx < lenZ; zIdx++)
        {
            Z[zIdx] = calcZAt(zIdx);
        }
    }

    assert(X[0] == XCutoffLower);
    //assert(X[lenX - 1] == XCutoffUpper);
    assert(Z[0] == ZCutoffLower);
    assert(Z[lenZ - 1] == ZCutoffUpper);
}

ZXGrid::~ZXGrid()
{
    delete[] X;
    delete[] Z;
}

double ZXGrid::getZAt(int ZIdx)
{
    return Z[ZIdx];
}

double ZXGrid::getXAt(int XIdx)
{
    return X[XIdx];
}
