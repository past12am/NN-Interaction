//
// Created by past12am on 8/25/23.
//

#ifndef NNINTERACTION_MOMENTUMLOOP_HPP
#define NNINTERACTION_MOMENTUMLOOP_HPP


#include "impulse/LoopImpulseGrid.hpp"

class MomentumLoop
{
    private:
        //LoopImpulseGrid loopImpulseGrid;
        int l2Points;
        int zPoints;
        int yPoints;
        int phiPoints;

    public:
        MomentumLoop(int l2Points, int zPoints, int yPoints, int phiPoints);
        virtual ~MomentumLoop();
};


#endif //NNINTERACTION_MOMENTUMLOOP_HPP
