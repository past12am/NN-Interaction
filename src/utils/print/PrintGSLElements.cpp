//
// Created by past12am on 8/26/23.
//

#include "../../../include/utils/print/PrintGSLElements.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>

const std::string PrintGSLElements::print_gsl_vector_complex(gsl_vector_complex* vec)
{
    std::stringstream sstream;
    for(size_t i = 0; i < vec->size; i++)
    {
        gsl_complex vec_element = gsl_vector_complex_get(vec, i);
        sstream << GSL_REAL(vec_element) << " " << ((GSL_IMAG(vec_element) < 0) ? "-" : "+") << "i " << abs(GSL_IMAG(vec_element)) << "   |   ";
    }

    return sstream.str();
}


const std::string PrintGSLElements::print_gsl_matrix_complex(gsl_matrix_complex* matrix)
{
    std::stringstream sstream;
    for(size_t i = 0; i < matrix->size1; i++)
    {
        sstream << "[";
        for(size_t j = 0; j < matrix->size2; j++)
        {
            gsl_complex mat_element = gsl_matrix_complex_get(matrix, i, j);

            sstream << std::setw(11) << std::setprecision(5) << GSL_REAL(mat_element) << " "
                    << ((GSL_IMAG(mat_element) < 0) ? "-" : "+") << "i "
                    << std::setw(11) << std::setprecision(5) << std::left << abs(GSL_IMAG(mat_element))
                    << std::right;
            if(j < matrix->size2 - 1) sstream << ", ";
        }
        sstream << "]" << std::endl;
    }

    return sstream.str();
}


const std::string PrintGSLElements::print_gsl_matrix_structure(gsl_matrix_complex* matrix, float eps)
{
    gsl_matrix* res = gsl_matrix_alloc(matrix->size1, matrix->size2);
    gsl_matrix_set_zero(res);

    for(size_t i = 0; i < matrix->size1; i++)
    {
        for (size_t j = 0; j < matrix->size2; j++)
        {
            gsl_complex entry = gsl_matrix_complex_get(matrix, i, j);

            if(abs(GSL_REAL(entry)) > eps || abs(GSL_IMAG(entry)) > eps)
                gsl_matrix_set(res, i, j, 1);
        }
    }



    std::stringstream sstream;
    for(size_t i = 0; i < matrix->size1; i++)
    {
        sstream << "[";
        for(size_t j = 0; j < matrix->size2; j++)
        {
            sstream << std::setw(2) << std::setprecision(1) << gsl_matrix_get(res, i, j);
            if(j < matrix->size2 - 1) sstream << ", ";
        }
        sstream << "]" << std::endl;
    }

    gsl_matrix_free(res);

    return sstream.str();
}