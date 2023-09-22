//
// Created by past12am on 8/26/23.
//

#ifndef NNINTERACTION_PRINTGSLELEMENTS_HPP
#define NNINTERACTION_PRINTGSLELEMENTS_HPP


#include <complex>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <string>

class PrintGSLElements
{
    public:
        static const std::string print_gsl_vector_complex(gsl_vector_complex* vec);
        static const std::string print_gsl_matrix_complex(gsl_matrix_complex* matrix);
};


#endif //NNINTERACTION_PRINTGSLELEMENTS_HPP
