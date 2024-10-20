//
// Created by past12am on 10/7/23.
//

#include <fstream>
#include <json/json.h>
#include <filesystem>

#include "../../include/scattering/ScatteringProcessHandler.hpp"
#include "../../include/scattering/processes/QuarkExchange.hpp"
#include "../../include/Definitions.h"
#include "../../include/scattering/processes/DiquarkExchange.hpp"

ScatteringProcessHandler::ScatteringProcessHandler(int numThreads, int lenX, int lenZ,
                                                   int k2Points, int zPoints, int yPoints, int phiPoints,
                                                   double eta,
                                                   double XCutoffLower, double XCutoffUpper,
                                                   double ZCutoffLower, double ZCutoffUpper,
                                                   const gsl_complex nucleonMass) :
    numThreads(numThreads), lenX(lenX), lenZ(lenZ), k2Points(k2Points), zPoints(zPoints), yPoints(yPoints),
    phiPoints(phiPoints), eta(eta), nucleon_mass(nucleonMass)
{
    subgridScatteringProcess = new ScatteringProcess*[numThreads];
    subgridIntegrationThread = new std::thread*[numThreads];



    ZXGrid completeZXGrid(lenX, lenZ, XCutoffLower, XCutoffUpper, ZCutoffLower, ZCutoffUpper);

    int numXPerThread = lenX / numThreads;
    if(lenX % numThreads != 0)
    {
        std::cout << "length of X grid must be divisible by number of threads" << std::endl;
        exit(-1);
    }


    for(int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        double curXCutoffLower = completeZXGrid.getXAt(threadIdx * numXPerThread);
        double curXCutoffUpper = completeZXGrid.getXAt((threadIdx + 1) * numXPerThread - 1);

        if(SCATTERING_PROCESS_TYPE == ScatteringProcessType::QUARK_EXCHANGE)
        {
            subgridScatteringProcess[threadIdx] = new QuarkExchange(numXPerThread, lenZ,
                                                                    curXCutoffLower, curXCutoffUpper,
                                                                    ZCutoffLower, ZCutoffUpper,
                                                                    nucleon_mass, eta,
                                                                    k2Points, zPoints, yPoints, phiPoints,
                                                                    threadIdx);
        }
        else if (SCATTERING_PROCESS_TYPE == ScatteringProcessType::DIQUARK_EXCHANGE)
        {
            subgridScatteringProcess[threadIdx] = new DiquarkExchange(numXPerThread, lenZ,
                                                                      curXCutoffLower, curXCutoffUpper,
                                                                      ZCutoffLower, ZCutoffUpper,
                                                                      nucleon_mass, eta,
                                                                      k2Points, zPoints, yPoints, phiPoints,
                                                                      threadIdx);
        }
        else
        {
            std::cout << "Could not determine Scattering Type" << std::endl;
            exit(-1);
        }
    }
}


ScatteringProcessHandler::~ScatteringProcessHandler()
{
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        delete subgridScatteringProcess[threadIdx];
    }

    delete[] subgridScatteringProcess;
}

void ScatteringProcessHandler::calculateScattering(double k2_cutoff)
{
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        subgridIntegrationThread[threadIdx] = new std::thread(&QuarkExchange::performScatteringCalculation,
                                                                  ((QuarkExchange*) subgridScatteringProcess[threadIdx]),
                                                                  k2_cutoff);
    }

    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
    {
        subgridIntegrationThread[threadIdx]->join();
    }
}

void ScatteringProcessHandler::store_scattering_amplitude(std::string data_path,
                                                          int lenX,
                                                          int lenZ,
                                                          double X_lower,
                                                          double X_upper,
                                                          double Z_lower,
                                                          double Z_upper,
                                                          double loop_cutoff,
                                                          int k2_integration_points,
                                                          int z_integration_points,
                                                          int y_integration_points,
                                                          int phi_integration_points)
{
    // Find/Create new specification-datapath
    //  BASE-b_I-x_DQ-y-z
    //          with x in {0, 1} ... Amplitude Isospin
    //               y in {scalar, axialvector} ... Diquark Type of Diquark 1
    //               z in {scalar, axialvector} ... Diquark Type of Diquark 2
    //               b in {tau, T} ... Basis Type (tau ... simple dirac basis, T ... Sym/Asym Basis)

    // Check for specification directory and create if needed
    std::ostringstream calc_spec_dir_strstream;
    calc_spec_dir_strstream << data_path << "/";
    calc_spec_dir_strstream << "BASE-" << BASIS << "_I-" << AMPLITUDE_ISOSPIN << "_DQ-" << DIQUARK_TYPE_1 << "-" << DIQUARK_TYPE_2 << "/";


    // Check for process specific directory and create if needed
    calc_spec_dir_strstream << SCATTERING_PROCESS_TYPE << "/";


    // Check for run directory and create new
    std::string calc_spec_dir_str = calc_spec_dir_strstream.str();
    std::string cur_run_dir = calc_spec_dir_str;
    if(std::filesystem::is_directory(calc_spec_dir_str))
    {
        // Find latest run
        int latest_run = 0;
        for (const auto & entry : std::filesystem::directory_iterator(calc_spec_dir_str))
        {
            if(!entry.is_directory())
                continue;

            std::string cur_entry_path = entry.path().string();

            int last_occ = cur_entry_path.find_last_of('_');
            int cur_latest_run = std::atoi(cur_entry_path.substr(last_occ + 1, cur_entry_path.length()).c_str());

            if(cur_latest_run > latest_run)
            {
                latest_run = cur_latest_run;
            }
        }

        cur_run_dir += "/run_" + std::to_string(latest_run + 1) + "/";
    }
    else
    {
        // Create directory and first run
        cur_run_dir += "/run_0/";
    }

    std::filesystem::create_directories(cur_run_dir);


    // Store dictionary with basic run specifications
    std::ostringstream specfnamestrstream;
    specfnamestrstream << cur_run_dir << "/spec.json";

    Json::Value spec_json_root;
    spec_json_root["basis"] = (std::ostringstream() << BASIS).str();
    spec_json_root["amplitude_isospin"] = AMPLITUDE_ISOSPIN;
    spec_json_root["diquark_type_1"] = (std::ostringstream() << DIQUARK_TYPE_1).str();
    spec_json_root["diquark_type_2"] = (std::ostringstream() << DIQUARK_TYPE_2).str();
    spec_json_root["X_points"] = std::to_string(lenX);
    spec_json_root["Z_points"] = std::to_string(lenZ);
    spec_json_root["X_range"] = (std::ostringstream() << "[" << X_lower << ", " << X_upper << "]").str();
    spec_json_root["Z_range"] = (std::ostringstream() << "[" << Z_lower << ", " << Z_upper << "]").str();
    spec_json_root["loop_cutoff"] = std::to_string(loop_cutoff);
    spec_json_root["k2_integration_points"] = std::to_string(k2_integration_points);
    spec_json_root["z_integration_points"] = std::to_string(z_integration_points);
    spec_json_root["y_integration_points"] = std::to_string(y_integration_points);
    spec_json_root["phi_integration_points"] = std::to_string(phi_integration_points);

    spec_json_root["projection_basis"] = (std::ostringstream() << PROJECTION_BASIS).str();
    spec_json_root["invert_strategy"] = (std::ostringstream() << INVERT_STRATEGY).str();

    std::ofstream spec_data_file;
    spec_data_file.open(specfnamestrstream.str(), std::ofstream::out | std::ios::trunc);
    spec_data_file << spec_json_root;
    spec_data_file.close();


    for(int basisElemIdx = 0; basisElemIdx < ((ScatteringProcess*) subgridScatteringProcess[0])->getTensorBasis()->getTensorBasisElementCount(); basisElemIdx++)
    {
        std::ostringstream fnamestrstream;
        fnamestrstream << cur_run_dir;

        if(BASIS == Basis::tau)
            fnamestrstream << "/tau_";
        else if(BASIS == Basis::T)
            fnamestrstream << "/T_";
        else if(BASIS == Basis::tau_prime)
            fnamestrstream << "/tauprime_";
        else
        {
            std::cout << "Unknown Basis " << BASIS << std::endl;
            exit(-1);
        }

        fnamestrstream << basisElemIdx << ".txt";


        std::ofstream data_file;
        data_file.open(fnamestrstream.str(), std::ofstream::out | std::ios::trunc);

        data_file << "X,Z,h,f,|scattering_amp|2" << std::endl;

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++)
        {
            ((ScatteringProcess*) subgridScatteringProcess[threadIdx])
                    ->store_scattering_amplitude(basisElemIdx, data_file);
        }
    }
}