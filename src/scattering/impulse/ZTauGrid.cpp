//
// Created by past12am on 10/7/23.
//

#include <cassert>
#include <gsl/gsl_complex_math.h>
#include "../../../include/scattering/impulse/ZTauGrid.hpp"


int ZTauGrid::getGridIdx(int tauIdx, int zIdx)
{
    return tauIdx * lenZ + zIdx;
}

int ZTauGrid::getLength() const
{
    return lenTau * lenZ;
}

double ZTauGrid::calcZAt(int zIdx)
{
    return zCutoffLower + (zCutoffUpper - zCutoffLower) * ((double) zIdx)/((double) (lenZ - 1));
}

double ZTauGrid::calcTauAt(int tauIdx)
{
    return tauCutoffLower + (tauCutoffUpper - tauCutoffLower) * ((double) tauIdx)/((double) (lenTau - 1));
}

int ZTauGrid::getLenTau() const
{
    return lenTau;
}

int ZTauGrid::getLenZ() const
{
    return lenZ;
}

ZTauGrid::ZTauGrid(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower,
                   double zCutoffUpper) : lenTau(lenTau), lenZ(lenZ),
                                                                  tauCutoffLower(tauCutoffLower),
                                                                  tauCutoffUpper(tauCutoffUpper),
                                                                  zCutoffLower(zCutoffLower),
                                                                  zCutoffUpper(zCutoffUpper)
{
    tau = new double[lenTau];
    z = new double[lenZ];

    for(int tauIdx = 0; tauIdx < lenTau; tauIdx++)
    {
        tau[tauIdx] = calcTauAt(tauIdx);

        for (int zIdx = 0; zIdx < lenZ; zIdx++)
        {
            z[zIdx] = calcZAt(zIdx);
        }
    }

    assert(tau[0] == tauCutoffLower);
    assert(tau[lenTau - 1] == tauCutoffUpper);
    assert(z[0] == zCutoffLower);
    assert(z[lenZ - 1] == zCutoffUpper);
}

ZTauGrid::~ZTauGrid()
{
    delete[] tau;
    delete[] z;
}

double ZTauGrid::getZAt(int zIdx)
{
    return z[zIdx];
}

double ZTauGrid::getTauAt(int tauIdx)
{
    return tau[tauIdx];
}
