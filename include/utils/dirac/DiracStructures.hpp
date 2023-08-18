//
// Created by past12am on 8/2/23.
//

#ifndef NNINTERACTION_DIRACSTRUCTURES_HPP
#define NNINTERACTION_DIRACSTRUCTURES_HPP

#include <gsl/gsl_matrix.h>



class DiracStructures
{
    public:
        DiracStructures();

        const gsl_matrix_complex* gamma5;
        const gsl_matrix_complex* gamma[];

        void slash(const gsl_vector_complex* p, gsl_matrix_complex* pSlash);
};


#endif //NNINTERACTION_DIRACSTRUCTURES_HPP
