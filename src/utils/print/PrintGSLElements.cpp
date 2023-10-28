//
// Created by past12am on 8/26/23.
//

#include "../../../include/utils/print/PrintGSLElements.hpp"

#include <iostream>
#include <sstream>

const std::string PrintGSLElements::print_gsl_vector_complex(gsl_vector_complex* vec)
{
    std::stringstream sstream;
    for(int i = 0; i < vec->size; i++)
    {
        gsl_complex vec_element = gsl_vector_complex_get(vec, i);
        sstream << GSL_REAL(vec_element) << " " << ((GSL_IMAG(vec_element) < 0) ? "-" : "+") << "i " << abs(GSL_IMAG(vec_element)) << "   |   ";
    }

    return sstream.str();
}


const std::string PrintGSLElements::print_gsl_matrix_complex(gsl_matrix_complex* matrix)
{
    std::stringstream sstream;
    for(int i = 0; i < matrix->size1; i++)
    {
        sstream << "[";
        for(int j = 0; j < matrix->size2; j++)
        {
            gsl_complex mat_element = gsl_matrix_complex_get(matrix, i, j);

            sstream << GSL_REAL(mat_element) << " " << ((GSL_IMAG(mat_element) < 0) ? "-" : "+") << "i " << abs(GSL_IMAG(mat_element));
            if(j < matrix->size2 - 1) sstream << ", ";
        }
        sstream << "]";
    }

    return sstream.str();
}