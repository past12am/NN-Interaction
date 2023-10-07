//
// Created by past12am on 10/7/23.
//

#ifndef NNINTERACTION_ZTAUGRID_HPP
#define NNINTERACTION_ZTAUGRID_HPP


class ZTauGrid
{
    protected:
        int lenTau;
        int lenZ;

        double tauCutoffLower;
        double tauCutoffUpper;

        double zCutoffLower;
        double zCutoffUpper;

        double* tau;
        double* z;


    public:
        ZTauGrid(int lenTau, int lenZ, double tauCutoffLower, double tauCutoffUpper, double zCutoffLower,
                 double zCutoffUpper);

        virtual ~ZTauGrid();

        double calcZAt(int zIdx);
        double calcTauAt(int tauIdx);

        double getZAt(int zIdx);
        double getTauAt(int tauIdx);

        int getGridIdx(int tauIdx, int zIdx);

        int getLenTau() const;
        int getLenZ() const;
        int getLength() const;
};


#endif //NNINTERACTION_ZTAUGRID_HPP
