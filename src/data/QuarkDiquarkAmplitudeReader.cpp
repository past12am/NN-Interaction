//
// Created by past12am on 30/05/24.
//

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include <gsl/gsl_complex_math.h>

#include "../../include/data/QuarkDiquarkAmplitudeReader.hpp"

#include <assert.h>

#include "../../include/gslhacks/GSLComplexOperators.hpp"

char* QuarkDiquarkAmplitudeReader::src_path = nullptr;
double QuarkDiquarkAmplitudeReader::c[8][4][3];

QuarkDiquarkAmplitudeReader* QuarkDiquarkAmplitudeReader::instance = nullptr;
QuarkDiquarkAmplitudeReader *QuarkDiquarkAmplitudeReader::getInstance(char* src_path)
{
    if(instance == nullptr)
    {
        instance = new QuarkDiquarkAmplitudeReader(src_path);
    }
    else
    {
        if(0 == std::strcmp(src_path, instance->getPath()))
        {
            return instance;
        }
        else
        {
            throw std::invalid_argument("Not implemented Yet - only supporting one QuarkDiquarkAmplitudeReader at a time");
        }
    }

    return instance;
}

QuarkDiquarkAmplitudeReader *QuarkDiquarkAmplitudeReader::getInstance()
{
    if(src_path != nullptr)
    {
        return getInstance(src_path);
    }
    else
    {
        throw::std::invalid_argument("Trying to create amplitude reader with no data file set");
    }
}

void QuarkDiquarkAmplitudeReader::setPath(char *src_path)
{
    QuarkDiquarkAmplitudeReader::src_path = src_path;
}

char *QuarkDiquarkAmplitudeReader::getPath()
{
    return src_path;
}

QuarkDiquarkAmplitudeReader::QuarkDiquarkAmplitudeReader(char *src_path)
{
    this->src_path = src_path;
    read_datafile();
}

gsl_complex QuarkDiquarkAmplitudeReader::f_k_cheby(gsl_complex p2, int amplitude_idx, int cheby_idx)
{
    //f = (c1 + c2*p^2) * e^(-c3*p^2).
    return (c[amplitude_idx][cheby_idx][0] + c[amplitude_idx][cheby_idx][1] * p2) * gsl_complex_exp(-c[amplitude_idx][cheby_idx][2] * p2);
}

gsl_complex QuarkDiquarkAmplitudeReader::f_k(gsl_complex p2, gsl_complex z, int amplitude_idx)
{
    gsl_complex p = gsl_complex_sqrt(p2);
    assert(abs(GSL_REAL(p)) > 0);
    assert(abs(GSL_IMAG(p)) < 1E-15);

    switch (amplitude_idx)
    {
        case 0:
        {
            gsl_complex fk_res0 = f_k_cheby(p2, amplitude_idx, 0);
            gsl_complex fk_res1 = f_k_cheby(p2, amplitude_idx, 1);
            gsl_complex fk_res2 = f_k_cheby(p2, amplitude_idx, 2);
            gsl_complex fk_res3 = f_k_cheby(p2, amplitude_idx, 3);

            // Ycomp(iAmp) = Y(iAmp,0) + Y(iAmp,1)*I*p*z + Y(iAmp,2)*p2*z**2 + Y(iAmp,3)*I*p*z**3
            gsl_complex res = fk_res0 + fk_res1 * gsl_complex_rect(0, 1) * p * z + fk_res2 * p2 * gsl_complex_pow_real(z, 2) + fk_res3 * gsl_complex_rect(0, 1) * p * gsl_complex_pow_real(z, 3);

            return res;
        }
        case 1:
        {
            gsl_complex fk_res0 = f_k_cheby(p2, amplitude_idx, 0);
            gsl_complex fk_res1 = f_k_cheby(p2, amplitude_idx, 1);
            gsl_complex fk_res2 = f_k_cheby(p2, amplitude_idx, 2);
            gsl_complex fk_res3 = f_k_cheby(p2, amplitude_idx, 3);

            // Ycomp(iAmp) = Y(iAmp,0) + Y(iAmp,1)*I*p*z + Y(iAmp,2)*z**2 + Y(iAmp,3)*I*p*z**3
            return  fk_res0 + fk_res1 * gsl_complex_rect(0, 1) * p * z + fk_res2 * gsl_complex_pow_real(z, 2) + fk_res3 * gsl_complex_rect(0, 1) * p * gsl_complex_pow_real(z, 3);
        }
    }

    throw std::invalid_argument("Not implmented yet - only supporting scalar quark-diquark amplitude");
    return gsl_complex();
}

void QuarkDiquarkAmplitudeReader::read_datafile()
{
    // Open file
    std::ifstream baryon_fit_reduced (src_path);
    std::string number;

    // skip first line
    baryon_fit_reduced.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int fitval_idx = 0;
    while(baryon_fit_reduced)
    {
        // skip preceeding whitespaces
        baryon_fit_reduced >> std::ws;


        // read blocks (amplitudes)
        for(int amplitude_idx = 0; amplitude_idx < 8; amplitude_idx++)
        {
            // read sections (chebys)
            for(int cheby_idx = 0; cheby_idx < 4; cheby_idx++)
            {
                baryon_fit_reduced >> number;
                c[amplitude_idx][cheby_idx][fitval_idx] = std::stod(number);

                // skip whitespaces after number
                baryon_fit_reduced >> std::ws;
            }
        }

        // proceed to next line
        fitval_idx++;
    }
}
