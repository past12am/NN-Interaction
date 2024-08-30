from os import listdir
from os.path import isfile, join, isdir

import shutil


NUM_BASIS_ELEM = 5

def find_latest_run_dir(run_dirs_path: str):
    latest_run_dir = None

    run_dir_idx = -1
    for run_directory in listdir(run_dirs_path):
        if(not isdir(run_dirs_path + run_directory)):
            continue

        cur_run_dir_idx = int(run_directory[run_directory.find("_")+1 : ])
        if(cur_run_dir_idx >= run_dir_idx):
            latest_run_dir = run_directory
            run_dir_idx = cur_run_dir_idx
    
    return latest_run_dir


def list_files_of_dir(path: str):
    return [f for f in listdir(path) if isfile(join(path, f))]

def get_files_with_substr_in_name(filelist, substr):
    return [i for i in filelist if substr in i]

def get_first_file_starting_with(filelist, substr):
    for fname in filelist:
        if fname[:len(substr)] == substr:
            return fname

def get_target_files_containing_substr(dir, substr):
    files = list_files_of_dir(dir)
    files_target = get_files_with_substr_in_name(files, substr)
    files_target_paths = [dir + f for f in files_target]

    return files_target_paths

def get_first_target_files_starting_with(dir, substrings):
    files = list_files_of_dir(dir)

    files_target_paths = list()
    for substr in substrings:
        file_target = get_first_file_starting_with(files, substr)
        files_target_paths.append(dir + file_target)

    return files_target_paths



# Build input path
base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/"

basis_type = "T"
invert_stategy = "numeric_matrix_inverse"

process_types = ["quark_exchange", "diquark_exchange"]
run_dirs = [None, None]

base_spec_dir_name = basis_type + "_" + invert_stategy



# Build Output Path
out_base = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/NNInteraction_Thesis/plots/"
amplitude_target_dir = out_base + "amplitudes/"
pwave_target_dir = out_base + "pwaves/"

qx_dir = "qx/"
dqx_dir = "dqx/"
combined_dir = "combined/"


qx_amp_filepaths = list()
dqx_amp_filepaths = list()
combined_amp_filepaths = list()

qx_pwave_filepaths = list()
dqx_pwave_filepaths = list()
combined_pwave_filepaths = list()



for pidx, process_type in enumerate(process_types):
    
    proc_run_base_path = base_path + "/" + base_spec_dir_name + "/" + process_type + "/"
    if (run_dirs[pidx] is None):
        run_path_name = find_latest_run_dir(proc_run_base_path)
        run_dirs[pidx] = run_path_name
    else:
        run_path_name = run_dirs[pidx]

    cur_path = proc_run_base_path + run_path_name + "/"


    # Process Parts
    for basis_idx in range(NUM_BASIS_ELEM):
        # T
        subdir = f"T-{basis_idx}"
        cur_subdir = cur_path + subdir + "/"
        target_files_T = get_target_files_containing_substr(cur_subdir, "_F_")
        

        # tau
        subdir = f"tau-{basis_idx}"
        cur_subdir = cur_path + subdir + "/"
        target_files_tau = get_target_files_containing_substr(cur_subdir, "_f_")


        # rho
        subdir = f"rho-{basis_idx}"
        cur_subdir = cur_path + subdir + "/"
        target_files_rho_amp = get_first_target_files_starting_with(cur_subdir, ["10", "32", "40", "41", "51"])
        target_files_rho_pwave = get_first_target_files_starting_with(cur_subdir, ["11", "22", "52", "60", "62"])


        if(process_type == "quark_exchange"):
            qx_amp_filepaths.extend(target_files_T)
            qx_amp_filepaths.extend(target_files_tau)
            qx_amp_filepaths.extend(target_files_rho_amp)

            qx_pwave_filepaths.extend(target_files_rho_pwave)

        elif(process_type == "diquark_exchange"):
            dqx_amp_filepaths.extend(target_files_T)
            dqx_amp_filepaths.extend(target_files_tau)
            dqx_amp_filepaths.extend(target_files_rho_amp)

            dqx_pwave_filepaths.extend(target_files_rho_pwave)
            
        else:
            raise Exception("Unknown Process Type")
        

        


# Combined Parts
comb_path = base_path + f"qx_{basis_type}_{invert_stategy}-dq_{basis_type}_{invert_stategy}/" + f"qx-{run_dirs[0]}_dqx-{run_dirs[1]}/"

target_files_comb_amp = get_target_files_containing_substr(comb_path, "FullSymAmplitude")
target_files_comb_pwave = get_target_files_containing_substr(comb_path, "Result")

combined_amp_filepaths.extend(target_files_comb_amp)
combined_pwave_filepaths.extend(target_files_comb_pwave)



# Copy files
#   QX
for amp_filepath in qx_amp_filepaths:
    shutil.copy2(amp_filepath, amplitude_target_dir + qx_dir)

for pwave_filepath in qx_pwave_filepaths:
    shutil.copy2(pwave_filepath, pwave_target_dir + qx_dir)


#   DX
for amp_filepath in dqx_amp_filepaths:
    shutil.copy2(amp_filepath, amplitude_target_dir + dqx_dir)

for pwave_filepath in dqx_pwave_filepaths:
    shutil.copy2(pwave_filepath, pwave_target_dir + dqx_dir)


#   Combined
for amp_filepath in combined_amp_filepaths:
    shutil.copy2(amp_filepath, amplitude_target_dir + combined_dir)

for pwave_filepath in combined_pwave_filepaths:
    shutil.copy2(pwave_filepath, pwave_target_dir + combined_dir)
