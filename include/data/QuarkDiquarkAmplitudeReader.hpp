//
// Created by past12am on 30/05/24.
//

#ifndef NNINTERACTION_QUARKDIQUARKAMPLITUDEREADER_HPP
#define NNINTERACTION_QUARKDIQUARKAMPLITUDEREADER_HPP



#include <gsl/gsl_complex.h>

class QuarkDiquarkAmplitudeReader
{
    private:
        static double c[8][4][3];
        static QuarkDiquarkAmplitudeReader* instance;
        static  char* src_path;

        QuarkDiquarkAmplitudeReader(char* src_path);

        void read_datafile();

    public:
        char* getPath();

        /*!
         *
         * @param p2
         * @param z
         * @param amplitude_idx     The index of the amplitude: 1, 2 --> Scalar Diquark Tensor 1 and 2,
         *                                                      3, .., 8 --> Axial Vector Diquark Tensor 1, ... 6
         * @return
         */
        gsl_complex f_k(gsl_complex p2, double z, int amplitude_idx);
        gsl_complex f_k(gsl_complex p2, int amplitude_idx, int cheby_idx);

        static void setPath(char* src_path);

        static QuarkDiquarkAmplitudeReader* getInstance(char* src_path);
        static QuarkDiquarkAmplitudeReader* getInstance();


};



#endif //NNINTERACTION_QUARKDIQUARKAMPLITUDEREADER_HPP
