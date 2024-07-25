import os
import json
import typing

import numpy as np
import pandas as pd

from basis.BasisTauToSymAsym import BasisTauToSymAsym


class Dataloader:

    def __init__(self, data_base_path: str, tensorbase_type: str, amplitude_isospin: int, process_type: str, dq_1_type: str, dq_2_type: str, Z_range, X_range_lower, run_nr: int = None) -> None:
        # Construct directory fitting specs
        data_path = data_base_path + f"/BASE-{tensorbase_type}_I-{amplitude_isospin}_DQ-{dq_1_type}-{dq_2_type}/"

        self.tensorbase_type = tensorbase_type
        self.amplitude_isospin = amplitude_isospin
        self.process_type = process_type
        self.dq_1_type = dq_1_type
        self.dq_2_type = dq_2_type


        # Check process data is accessible
        data_path_process = data_path + f"/{process_type}/"
        if(not os.path.isdir(data_path_process)):
            exit(-1)


        # Find latest run
        if(run_nr is None):
            self.latest_run_dir_process = Dataloader.find_latest_run_dir(data_path_process)
        else:
            self.latest_run_dir_process = f"run_{run_nr}"

        if(self.latest_run_dir_process is None):
            exit(-1)

        data_path_process_latest = data_path_process + self.latest_run_dir_process + "/"
        self.data_path = data_path_process_latest


        # Check which basis we are working in
        spec_file = open(data_path_process_latest + "/spec.json")
        self.process_spec = json.load(spec_file)
        spec_file.close()


        # Sanity checks, specified base, dq type and so on should match
        if(not self.process_spec["basis"] == tensorbase_type or 
           not self.process_spec["amplitude_isospin"] == amplitude_isospin or 
           not self.process_spec["diquark_type_1"] == dq_1_type or 
           not self.process_spec["diquark_type_2"] == dq_2_type):
            exit(-1)



        # Load Dirac Space form factors and independent variables of process
        self.pd_dirac_space_form_factors_list = Dataloader.load_dirac_space_form_factors(data_path_process_latest, tensorbase_type, Z_range, X_range_lower)
        
        # Load impulse grid
        self.X, self.Z, self.X_unique, self.Z_unique = Dataloader.build_XZ_array_from_pd(self.pd_dirac_space_form_factors_list[0])

        # Load h functions
        self.h = Dataloader.build_matrix_from_pd_list(self.pd_dirac_space_form_factors_list, int(self.process_spec["X_points"]), int(self.process_spec["Z_points"]), "h")

        # Base Transformation to other base in case we use the tau base
        if (self.tensorbase_type == "tau"):
            self.f = Dataloader.build_matrix_from_pd_list(self.pd_dirac_space_form_factors_list, int(self.process_spec["X_points"]), int(self.process_spec["Z_points"]), "f")
            self.F = BasisTauToSymAsym.build_alternate_basis_numpy(self.f)
        
        elif (self.tensorbase_type == "T"):
            self.F = Dataloader.build_matrix_from_pd_list(self.pd_dirac_space_form_factors_list, int(self.process_spec["X_points"]), int(self.process_spec["Z_points"]), "f")
            self.f = BasisTauToSymAsym.build_alternate_basis_inverse_numpy(self.F)



        # Switch sign of amplitudes for diquark exchange
        if(process_type == "quark_exchange"):
            self.F *= -1
            self.f *= -1

        # Load flavor factors
        self.process_flavor_factor = Dataloader.load_flavor_space_form_factors("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/flavorspace", process_type, amplitude_isospin, dq_1_type, dq_2_type)

        # Build flavored dressing functions
        #self.f_flavored = self.f * self.process_flavor_factor
        #self.F_flavored = self.F * self.process_flavor_factor

        #self.f = self.f_flavored
        #self.F = self.F_flavored    # TODO check this --> we need the flavored ones to correctly reproduce sym/anti-sym


        # TODO color factor
        


    
    @staticmethod
    def build_matrix_from_pd_list(pd_scattering_list: typing.List, lenX: int, lenZ: int, colname: str):
        col_vals_matrix = np.zeros((len(pd_scattering_list), len(pd.unique(pd_scattering_list[0]["X"])), len(pd.unique(pd_scattering_list[0]["Z"]))))

        for basis_idx, pd_scattering in enumerate(pd_scattering_list):
            pd_scattering = pd_scattering.sort_values(["X", "Z"], ascending=True)

            X_idx = -1
            Z_idx = -1

            last_Z = -5
            last_X = -5
            for _, row in pd_scattering.iterrows():
                if(row["X"] > last_X):
                    X_idx += 1
                    last_X = row["X"]

                if(row["Z"] > last_Z):
                    Z_idx += 1
                    last_Z = row["Z"]

                elif(row["Z"] < last_Z):
                    Z_idx = 0
                    last_Z = row["Z"]

                col_vals_matrix[basis_idx, X_idx, Z_idx] = np.real(row[colname])

        return col_vals_matrix




    @staticmethod
    def build_XZ_array_from_pd(pd_scattering: pd.DataFrame):
        X_vals = pd_scattering["X"].to_numpy()
        Z_vals = pd_scattering["Z"].to_numpy()

        X_vals_unique = pd_scattering["X"].unique()
        Z_vals_unique = pd_scattering["Z"].unique()

        return X_vals, Z_vals, X_vals_unique, Z_vals_unique


    @DeprecationWarning
    def restrict_dataset(self, pd_ff, Z_range, X_range_lower):
        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_ff["Z"])) < Z_range, np.array(pd_ff["X"]) > X_range_lower))[0]
        pd_ff = pd_ff.iloc[idx_selector]

        return pd_ff


    @staticmethod
    def find_latest_run_dir(run_dirs_path: str):
        latest_run_dir = None

        run_dir_idx = -1
        for run_directory in os.listdir(run_dirs_path):
            if(not os.path.isdir(run_dirs_path + run_directory)):
                continue

            cur_run_dir_idx = int(run_directory[run_directory.find("_")+1 : ])
            if(cur_run_dir_idx >= run_dir_idx):
                latest_run_dir = run_directory
                run_dir_idx = cur_run_dir_idx
        
        return latest_run_dir
    

    @staticmethod
    def load_flavor_space_form_factors(flavor_data_path: str, process_type, amplitude_isospin, dq_1_type, dq_2_type):
        pd_process_flavor = pd.read_csv(flavor_data_path + f"/{process_type}.csv", delimiter=";", decimal=".")

        process_flavor_factor = pd_process_flavor[(pd_process_flavor.amplitude_isospin == amplitude_isospin) &
                                                  (pd_process_flavor.diquark_1_type == dq_1_type) &
                                                  (pd_process_flavor.diquark_2_type == dq_2_type)]["flavor_factor"].values[0]

        try:
            process_flavor_factor = float(process_flavor_factor)
        except:
            a , b = process_flavor_factor.split("/")
            process_flavor_factor = int(a) / int(b)


        return process_flavor_factor
    

    @staticmethod
    def load_dirac_space_form_factors(data_path: str, tensorbase_type: str, Z_range, X_range_lower):

        pd_dirac_form_factors = list()
        for baseElemIdx in range(5):
            pd_ff = pd.read_csv(data_path + f"/{tensorbase_type}_{baseElemIdx}.txt")
            pd_ff = pd_ff.map(lambda s: complex(s.replace('i', 'j')) if(isinstance(s, str)) else s)

            idx_selector = np.where(np.logical_and(np.abs(np.array(pd_ff["Z"])) <= Z_range, np.array(pd_ff["X"]) >= X_range_lower))[0]
            pd_ff = pd_ff.iloc[idx_selector]

            pd_dirac_form_factors.append(pd_ff)

        return pd_dirac_form_factors