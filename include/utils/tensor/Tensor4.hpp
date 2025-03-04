//
// Created by past12am on 8/9/23.
//

#ifndef NNINTERACTION_TENSOR4_HPP
#define NNINTERACTION_TENSOR4_HPP

#include <complex>
#include <gsl/gsl_matrix.h>
#include <ostream>
#include <cassert>
#include <iostream>
#include "gsl/gsl_complex_math.h"

template<int d1, int d2, int d3, int d4> class Tensor4
{
    private:
        gsl_complex tensor[d1][d2][d3][d4];

    public:
        /*!
         * Creates a Tensor from an outer product between A and B
         * @param A
         * @param B
         */
        Tensor4(gsl_matrix_complex* A, gsl_matrix_complex* B)
        {
            for(size_t i = 0; i < A->size1; i++)
            {
                for(size_t j = 0; j < A->size2; j++)
                {
                    for(size_t k = 0; k < B->size1; k++)
                    {
                        for(size_t l = 0; l < B->size2; l++)
                        {
                            tensor[i][j][k][l] = gsl_complex_mul(gsl_matrix_complex_get(A, i, j), gsl_matrix_complex_get(B, k, l));
                        }
                    }
                }
            }
        }

        Tensor4()
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = gsl_complex_rect(0, 0);
                        }
                    }
                }
            }
        }

        Tensor4(Tensor4 &tensor4)
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = tensor4.tensor[i][j][k][l];
                        }
                    }
                }
            }
        }

        void setElement(size_t i, size_t j, size_t k, size_t l, gsl_complex x)
        {
            tensor[i][j][k][l] = x;
        }

        friend std::ostream &operator<<(std::ostream &os, const Tensor4 &tensor4)
        {
            os << "tensor4: " << tensor4.tensor;
            return os;
        }

        gsl_complex leftContractWith(Tensor4<4, 4, 4, 4>* other)
        {
            gsl_complex res = gsl_complex_rect(0, 0);

            for(int alpha = 0; alpha < d1; alpha++)
            {
                for (int beta = 0; beta < d2; beta++)
                {
                    for (int gamma = 0; gamma < d3; gamma++)
                    {
                        for (int delta = 0; delta < d4; delta++)
                        {
                            res = gsl_complex_add(res, gsl_complex_mul( other->tensor[beta][alpha][delta][gamma], tensor[alpha][beta][gamma][delta]));

                            //Out of interest
                            //res = gsl_complex_add(res, gsl_complex_mul( other->tensor[delta][gamma][beta][alpha], tensor[alpha][beta][gamma][delta]));
                        }
                    }
                }
            }

            return res;
        }

        Tensor4<d1, d2, d3, d4> operator*(const gsl_complex& scalar)
        {
            Tensor4<d1, d2, d3, d4> res = Tensor4<d1, d2, d3, d4>();
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            res.tensor[i][j][k][l] = gsl_complex_mul(tensor[i][j][k][l], scalar);
                        }
                    }
                }
            }

            return res;
        }

        Tensor4<d1, d2, d3, d4>& operator+=(const Tensor4<d1, d2, d3, d4>& other)
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = gsl_complex_add(tensor[i][j][k][l], other.tensor[i][j][k][l]);
                        }
                    }
                }
            }

            return *this;
        }

        Tensor4<d1, d2, d3, d4> operator+(const Tensor4<d1, d2, d3, d4>& other)
        {
            Tensor4<d1, d2, d3, d4> res = Tensor4<d1, d2, d3, d4>();
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            res.setElement(i, j, k, l, gsl_complex_add(tensor[i][j][k][l], other.tensor[i][j][k][l]));
                        }
                    }
                }
            }

            return res;
        }

        Tensor4<d1, d2, d3, d4> operator-(const Tensor4<d1, d2, d3, d4>& other)
        {
            Tensor4<d1, d2, d3, d4> res = Tensor4<d1, d2, d3, d4>();
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            res.setElement(i, j, k, l, gsl_complex_sub(tensor[i][j][k][l], other.tensor[i][j][k][l]));
                        }
                    }
                }
            }

            return res;
        }

        double absSquare()
        {
            gsl_complex res = gsl_complex_rect(0, 0);
            //gsl_complex res2 = gsl_complex_rect(0, 0);

            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            gsl_complex comp_res = gsl_complex_mul(tensor[i][j][k][l], gsl_complex_conjugate(tensor[j][i][l][k]));
                            //gsl_complex comp_res2 = gsl_complex_mul(tensor[i][j][k][l], gsl_complex_conjugate(tensor[l][k][j][i]));
                            // M_alpha,beta;gamma,delta . (M_beta,alpha;delta,gamma)*

                            res = gsl_complex_add(res, comp_res);
                            //res2 = gsl_complex_add(res2, comp_res2);
                        }
                    }
                }
            }

            //std::cout << GSL_REAL(res) << " + i " << GSL_IMAG(res) << "   #   " << GSL_REAL(res2) << " + i " << GSL_IMAG(res2) << std::endl;
            //assert(GSL_IMAG(res) < 1E-15);

            //assert(GSL_REAL(res) == GSL_REAL(res2));
            //assert(GSL_IMAG(res) == GSL_IMAG(res2));

            return GSL_REAL(res);
        }

        void setZero()
        {
            for(int i = 0; i < d1; i++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int k = 0; k < d3; k++)
                    {
                        for (int l = 0; l < d4; l++)
                        {
                            tensor[i][j][k][l] = GSL_COMPLEX_ZERO;
                        }
                    }
                }
            }
        }
};

#endif //NNINTERACTION_TENSOR4_HPP
