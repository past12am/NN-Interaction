//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_PROJECTORS_HPP
#define NNINTERACTION_PROJECTORS_HPP

#include <complex>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class Projectors
{
    private:
        static const gsl_matrix_complex* unitM;

        static bool checkProjectorProperties(gsl_matrix_complex* projector);

    public:
        static void posEnergyProjector(gsl_vector_complex* P, gsl_matrix_complex* posEnergyProj);
        static void transverseProjector(gsl_vector_complex* P, gsl_matrix_complex* transvProj);
        static void longitudinalProjector(gsl_vector_complex* P, gsl_matrix_complex* longitudProj);

        static const gsl_matrix_complex* getUnitM();
};


#endif //NNINTERACTION_PROJECTORS_HPP
