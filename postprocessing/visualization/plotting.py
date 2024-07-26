import os
import typing
import json

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from data.AmplitudeHandler import AmplitudeHandler


pwave_names = ["s", "p", "d", "f", "g", "h", "i", "j", "k"]


class PlotterFullAmplitude:

    def __init__(self, base_path: str, dataloader_qx, dataloader_dqx, savefig) -> None:
        self.base_path = base_path

        self.dataloader_qx = dataloader_qx
        self.dataloader_dqx = dataloader_dqx

        self.run_path_name = f"qx-{dataloader_qx.latest_run_dir_process}_dqx-{dataloader_dqx.latest_run_dir_process}"

        self.savefig = savefig
        self.show_plots = True


        self.base_spec_dir_name = "qx_" + dataloader_qx.process_spec["basis"] + "_" + dataloader_qx.process_spec["invert_strategy"] + "-dq_" + dataloader_dqx.process_spec["basis"] + "_" + dataloader_dqx.process_spec["invert_strategy"]


        # Construct current working path
        self.cur_proc_run_base_path = self.base_path + "/" + self.base_spec_dir_name + "/" + self.run_path_name + "/"

        # Check if paths exist or create
        if(self.savefig):
            os.makedirs(self.cur_proc_run_base_path, exist_ok=True)

            # Copy Spec dict to output directory
            json.dump({"quark-exchange": dataloader_qx.process_spec, "diquark-exchange": dataloader_dqx.process_spec}, open(self.cur_proc_run_base_path + "specCombined.json", 'w'))
    

    def save_active_fig(self, fig_name):
        plt.savefig(self.cur_proc_run_base_path + "/" + f"{fig_name}.png", dpi=600)


    def plotFullSymAmplitudeIsospin0(self, tensor_basis_names, fig_name):
        # Plot Full (Anti-) Symmetric Amplitude
        F_complete_I0 = 1/2 * (self.dataloader_dqx.F - self.dataloader_qx.F)
        self.plot_full_amplitude_np(self.dataloader_qx.X, self.dataloader_qx.Z, F_complete_I0, tensor_basis_names, isospin=0, fig_name=fig_name)

    def plotFullSymAmplitudeIsospin1(self, tensor_basis_names, fig_name):
        # Plot Full (Anti-) Symmetric Amplitude
        F_complete_I1 = 1/2 * (self.dataloader_dqx.F + self.dataloader_qx.F)
        self.plot_full_amplitude_np(self.dataloader_qx.X, self.dataloader_qx.Z, F_complete_I1, tensor_basis_names, isospin=1, fig_name=fig_name)


    def plot_full_amplitude_np(self, X: np.ndarray, Z: np.ndarray, F: np.ndarray, tensor_basis_names, isospin: int, fig_name: str):
        fig = plt.figure(figsize=(7, 9))
        fig.tight_layout()

        axs = [
            plt.subplot2grid(shape=(3,4), loc=(0,0), colspan=2, projection='3d'),
            plt.subplot2grid(shape=(3,4), loc=(0,2), colspan=2, projection='3d'),
            plt.subplot2grid(shape=(3,4), loc=(1,0), colspan=2, projection='3d'),
            plt.subplot2grid(shape=(3,4), loc=(1,2), colspan=2, projection='3d'),
            plt.subplot2grid(shape=(3,4), loc=(2,1), colspan=2, projection='3d'),
        ]

        for basis_idx in range(F.shape[0]):
            ax = axs[basis_idx]
            ax.ticklabel_format(style='plain')
            ax.plot_trisurf(X, Z, F[basis_idx, :, :].flatten(), cmap=cm.coolwarm)
            ax.set_xlabel("$X$")
            ax.set_ylabel("$Z$")

            if(basis_idx < 2):
                ax.set_title(tensor_basis_names[basis_idx] + f": $f^{{({isospin})}}_{basis_idx + 1}(X, Z)$")
                ax.set_zlabel(f"$f^{{({isospin})}}_{basis_idx + 1}$", labelpad=10)
            else:
                ax.set_title(tensor_basis_names[basis_idx] + f": $g^{{({isospin})}}_{basis_idx + 1 - 2}(X, Z)$")
                ax.set_zlabel(f"$g^{{({isospin})}}_{basis_idx + 1 - 2}$", labelpad=10)
            ax.set_ylim([-1, 1])
            ax.zaxis.set_rotate_label(False)


        fig.subplots_adjust(wspace=0, hspace=0.385, top=0.94, bottom=0.04, left=0.01, right=0.945)

        if(self.savefig):
            self.save_active_fig(fig_name)

        if(self.show_plots):
            plt.show()

        plt.close()


    def plot_final_res(self, f_l, x, xlabel, x_label_unit, ylabels, tensorBasisNamesDict, exchange_channel, xmax):
        for basis_idx in range(f_l.shape[0]):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            ax.set_xlim(right=xmax)

            for l in range(f_l.shape[1]):
                ax.plot(x, f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
            
            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.155, right=0.92, hspace=0.2, wspace=0.2)

            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(tensorBasisNamesDict["rho"][basis_idx], x=mid, fontsize="x-large")
            
            ax.set_title(f"${ylabels[basis_idx]}({xlabel})$")
            ax.set_xlabel(f"${xlabel} \\ {x_label_unit}$", fontsize="large")
            ax.set_ylabel(f"${ylabels[basis_idx]}$", fontsize="large")
            ax.grid(color='lightgray', linestyle='dashed')
            ax.legend()

            if(self.savefig):
                self.save_active_fig(f"Result_{exchange_channel}_rho{basis_idx}")

            if(self.show_plots):
                plt.show()

            plt.close()




class Plotter:

    def __init__(self, base_path: str, tensorBasisNamesDict, process_type, run_path_name, run_spec_dict, savefig, include_loglog_plots: bool=True) -> None:
        self.base_path = base_path

        self.tensorBasisNamesDict = tensorBasisNamesDict

        self.run_spec_dict = run_spec_dict
        self.base_spec_dir_name = run_spec_dict["basis"] + "_" + run_spec_dict["invert_strategy"]

        self.process_type = process_type
        self.run_path_name = run_path_name

        self.savefig = savefig
        self.show_plots = True

        self.include_loglog_plots = include_loglog_plots


        # Construct current working path
        self.cur_proc_run_base_path = self.base_path + "/" + self.base_spec_dir_name + "/" + self.process_type + "/" + self.run_path_name + "/"

        # Check if paths exist or create
        if(self.savefig):
            os.makedirs(self.cur_proc_run_base_path, exist_ok=True)

            # Copy Spec dict to output directory
            json.dump(self.run_spec_dict, open(self.cur_proc_run_base_path + "spec.json", 'w'))

            for basis_idx in range(5):
                os.makedirs(self.base_path_for("T", basis_idx), exist_ok=True) 
                os.makedirs(self.base_path_for("tau", basis_idx), exist_ok=True) 
                os.makedirs(self.base_path_for("tau_prime", basis_idx), exist_ok=True) 
                os.makedirs(self.base_path_for("rho", basis_idx), exist_ok=True) 

            os.makedirs(self.base_path_for("generic"), exist_ok=True)


    def base_path_for(self, base_type, basis_idx: int=None):
        if (basis_idx is not None):
            return self.cur_proc_run_base_path + f"/{base_type}-{basis_idx}/"
        else:
            return self.cur_proc_run_base_path + f"/{base_type}/"
    

    def save_active_fig(self, fig_name, step_idx, base_type, basis_idx: int=None):
        plt.savefig(self.base_path_for(base_type, basis_idx) + "/" + f"{step_idx:02d}__{fig_name}_{base_type}_{basis_idx + 1}.png", dpi=600)


    def plot_form_factor_np(self, X: np.ndarray, Z: np.ndarray, dressing_f: np.ndarray, dressing_f_name: str, tensor_basis_elem: str, base_type: str, basis_idx: int, fig_name: str, step_idx: int, do_title: bool=True):
        fig = plt.figure(figsize=(10, 5))

        if do_title:
            fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

        # Subplot real
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(f"${dressing_f_name}(X, Z)$")
        ax.plot_trisurf(X, Z, np.real(dressing_f.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"${dressing_f_name}$", labelpad=10)
        ax.set_ylim([-1, 1])
        ax.zaxis.set_rotate_label(False)


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, base_type, basis_idx)

        if(self.show_plots):
            plt.show()

        plt.close()

    
    def plot_form_factor_np_with_imag(self, X: np.ndarray, Z: np.ndarray, dressing_f: np.ndarray, dressing_f_name: str, tensor_basis_elem: str, base_type: str, basis_idx: int, fig_name: str, step_idx: int):
        fig = plt.figure(figsize=(10, 5))

        fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

        # Subplot real
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title(f"$\\Re({dressing_f_name}(X, Z))$")
        ax.plot_trisurf(X, Z, np.real(dressing_f.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Re({dressing_f_name})$")


        # Subplot imag
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title(f"$\\Im({dressing_f_name}(X, Z))$")
        ax.plot_trisurf(X, Z, np.imag(dressing_f.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Im({dressing_f_name})$")


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, base_type, basis_idx)

        if(self.show_plots):
            plt.show()

        plt.close()



    def plot_form_factor_np_side_by_side(self, X1: np.ndarray, Z1: np.ndarray, dressing_f1: np.ndarray, dressing_f_name1: str, xlabel1: str,
                                        X2: np.ndarray, Z2: np.ndarray, dressing_f2: np.ndarray, dressing_f_name2: str, xlabel2: str, 
                                        tensor_basis_elem: str, basis_idx: int, base_type: str, fig_name: str, step_idx: int, left_pretitle: str = None, right_pretitle: str = None):
        dressing_f1_params = f"({xlabel1}, Z)"
        dressing_f2_params = f"({xlabel2}, Z)"

        left_title = f"${dressing_f_name1}_{basis_idx + 1}{dressing_f1_params}$"
        right_title = f"${dressing_f_name2}_{basis_idx + 1}{dressing_f2_params}$"

        if (left_pretitle is not None):
            left_title = left_pretitle + left_title

        if (right_pretitle is not None):
            right_title = right_pretitle + right_title

        fig = plt.figure(figsize=(10, 5))
        fig.tight_layout()

        fig.suptitle("Tensor Basis Element " + tensor_basis_elem)


        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title(left_title)
        ax.plot_trisurf(X1, Z1, np.real(dressing_f1.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel1}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"${dressing_f_name1}_{basis_idx + 1}$", labelpad=8)
        ax.set_ylim([-1, 1])
        ax.zaxis.set_rotate_label(False)


        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title(right_title)
        ax.plot_trisurf(X2, Z2, np.real(dressing_f2.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel2}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"${dressing_f_name2}_{basis_idx + 1}$", labelpad=14)
        ax.set_ylim([-1, 1])
        ax.zaxis.set_rotate_label(False)

        fig.subplots_adjust(top=0.88, bottom=0.11, left=0.01, right=0.91, hspace=0.2, wspace=0.15)


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, base_type, basis_idx)
        if(self.show_plots):
            plt.show()
        plt.close()


    def plot_form_factor_np_side_by_side_with_imag(self, X1: np.ndarray, Z1: np.ndarray, dressing_f1: np.ndarray, dressing_f_name1: str, xlabel1: str,
                                        X2: np.ndarray, Z2: np.ndarray, dressing_f2: np.ndarray, dressing_f_name2: str, xlabel2: str, 
                                        tensor_basis_elem: str, basis_idx: int, base_type: str, fig_name: str, step_idx: int):
        dressing_f1_params = f"({xlabel1}, Z)"
        dressing_f2_params = f"({xlabel2}, Z)"

        fig = plt.figure(figsize=(10, 9))

        fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

        # Subplot real
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_title(f"$\\Re({dressing_f_name1}{dressing_f1_params})$")
        ax.plot_trisurf(X1, Z1, np.real(dressing_f1.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel1}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Re({dressing_f_name1})$")


        # Subplot imag
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title(f"$\\Im({dressing_f_name1}{dressing_f1_params})$")
        ax.plot_trisurf(X1, Z1, np.imag(dressing_f1.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel1}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Im({dressing_f_name1})$")



        # Subplot real
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title(f"$\\Re({dressing_f_name2}{dressing_f2_params})$")
        ax.plot_trisurf(X2, Z2, np.real(dressing_f2.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel2}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Re({dressing_f_name2})$")


        # Subplot imag
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title(f"$\\Im({dressing_f_name2}{dressing_f2_params})$")
        ax.plot_trisurf(X2, Z2, np.imag(dressing_f2.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel2}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\\Im({dressing_f_name2})$")


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, base_type, basis_idx)
        if(self.show_plots):
            plt.show()
        plt.close()



    def plotAmplitudes(self, dataloader, fig_name_f, fig_name_F, step_idx: int, process_abbrev: str):
        # process_abbrev = q (Quark Exchange), dq (Diquark Exchange)
        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.f[base_idx, :, :], f"f^{{({process_abbrev})}}_{base_idx + 1}", f"$\\tau_{base_idx + 1} = $ " + self.tensorBasisNamesDict["tau"][base_idx], "tau", base_idx, fig_name=f"{fig_name_f}_{base_idx + 1}", step_idx=step_idx)

        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.F[base_idx, :, :], f"F^{{({process_abbrev})}}_{base_idx + 1}", f"$T_{base_idx + 1} = $ " + self.tensorBasisNamesDict["T"][base_idx], "T", base_idx, fig_name=f"{fig_name_F}_{base_idx + 1}", step_idx=step_idx)


    def plotAmplitudes_h(self, dataloader, fig_name_h, projection_basis_type: str, step_idx: int, process_abbrev: str):
        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.h[base_idx, :, :], f"h^{{({process_abbrev})}}_{base_idx + 1}", self.tensorBasisNamesDict[projection_basis_type][base_idx], projection_basis_type, base_idx, fig_name=f"{fig_name_h}_{base_idx + 1}", step_idx=step_idx, do_title=False)


    def plotAmplitudesPartialWaveExpandedAndOriginal(self, grid_var1: np.ndarray, Z: np.ndarray, V_l: np.ndarray, V: np.ndarray, var1_name: str, fig_name, step_idx: int):
        # Build original amplitudes from partial wave expanded ones
        num_Z_check = 21
        Z_check_linspace = np.linspace(np.min(Z), np.max(Z), num_Z_check)

        V_qx_check = np.zeros((V.shape[0], V.shape[1], num_Z_check))

        for basis_idx in range(V_l.shape[0]):
            for X_idx in range(V_l.shape[2]):
                #V_check = np.polynomial.Legendre(V_qx_l[basis_idx, :, X_idx], domain=[-1, 1])
            
                V_qx_check[basis_idx, X_idx, :] = np.polynomial.legendre.legval(Z_check_linspace, V_l[basis_idx, :, X_idx])

            X_qx_extended = np.repeat(grid_var1, len(Z))
            Z_qx_extended = np.tile(Z, len(grid_var1))

            X_check_extended = np.repeat(grid_var1, num_Z_check)
            Z_check_linspace_extended = np.tile(Z_check_linspace, len(grid_var1))

            self.plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V[basis_idx, :, :], "U", var1_name,
                                                  X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "U^{{(\\mathrm{{check}})}}", var1_name,
                                                  self.tensorBasisNamesDict["rho"][basis_idx], basis_idx, "rho", fig_name=fig_name, step_idx=step_idx)                 # f"Amplitude_rho_{basis_idx + 1}"
            

    def plot_pwave_amp_with_fits(self, V_qx_l, X_qx, ampHandler: AmplitudeHandler, base_type, fig_name: str, step_idx: int):
        for basis_idx in range(V_qx_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axs = [ax]

            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.155, right=0.92, hspace=0.2, wspace=0.2)

            for l in range(V_qx_l.shape[1]):

                fit_vals = np.zeros_like(X_qx)
                for X_idx, X in enumerate(X_qx):
                    fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)


                axs[0].plot(X_qx, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                axs[0].plot(X_qx, V_qx_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")
                axs[0].grid(color='lightgray', linestyle='dashed')

                if(self.include_loglog_plots):
                    axs[1].loglog(X_qx, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                    axs[1].loglog(X_qx, V_qx_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")

            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx], x=mid)

            axs[0].set_title(f"$U^{{(l)}}_{basis_idx + 1}(X)$", fontsize="large")
            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel(f"$U^{{(l)}}_{basis_idx + 1}$", fontsize="large")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log X$")
                axs[1].set_ylabel("$\\log V_{l}(X)$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) # "fits-V_l(X)"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp(self, f_l, x, xlabel, x_label_unit, fig_name, base_type, step_idx: int, max_wave: int=None):
        for basis_idx in range(f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axs = [ax]

            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.155, right=0.92, hspace=0.2, wspace=0.2)

            for l in range(f_l.shape[1] if max_wave is None else max_wave):
                axs[0].plot(x, f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")

                if(self.include_loglog_plots):
                    axs[1].loglog(x, f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                
            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx], x=mid, fontsize="x-large")

            axs[0].set_title(f"$U_{basis_idx + 1}^{{(l)}}({xlabel})$")
            axs[0].set_xlabel(f"${xlabel} \\setminus {x_label_unit}$", fontsize="large")
            axs[0].set_ylabel(f"$U_{basis_idx + 1}^{{(l)}}$", fontsize="large")
            axs[0].grid(color='lightgray', linestyle='dashed')
            #axs[0].spines[['right', 'top']].set_visible(False)
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel(f"$\\log {xlabel}  {x_label_unit}$")
                axs[1].set_ylabel(f"$\\log V_l({xlabel})$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #f"V_l({xlabel})"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_scaled_side_by_side(self, f_l, x, xlabel, x_label_unit, fig_name, base_type, step_idx: int, y_lim: typing.Tuple=None, max_wave: int=None):
        for basis_idx in range(f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(11, 6))

            for l in range(f_l.shape[1] if max_wave is None else max_wave):
                axs[0].plot(x, f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                axs[1].plot(x, f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                
            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.1, right=0.95, hspace=0.2, wspace=0.3)

            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx], x=mid)

            axs[0].set_title(f"$U_{basis_idx + 1}^{{(l)}}({xlabel})$")
            axs[0].set_xlabel(f"${xlabel} \\ {x_label_unit}$", fontsize="large")
            axs[0].set_ylabel(f"$U_{basis_idx + 1}^{{(l)}}$", fontsize="large")
            axs[0].grid(color='lightgray', linestyle='dashed')

            axs[1].set_title(f"$U_{basis_idx + 1}^{{(l)}}({xlabel})$")
            axs[1].set_xlabel(f"${xlabel} \\ {x_label_unit}$", fontsize="large")
            axs[1].set_ylabel(f"$U_{basis_idx + 1}^{{(l)}}$", fontsize="large")
            axs[1].grid(color='lightgray', linestyle='dashed')

            axs[0].legend()
            axs[1].legend()

            if(y_lim is not None):
                axs[1].set_ylim(y_lim[basis_idx])

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx)    # f"scale-sidebyside-V_l({xlabel})"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_wave_sum(self, f_l, x, xlabel, x_label_unit, fig_name, base_type, step_idx: int, max_wave: int=None):
        for basis_idx in range(f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axs = [ax]

            f_l_r_sum = np.sum(f_l, axis=1)
            
            axs[0].plot(x, f_l_r_sum[basis_idx, :], label="Sum partial waves")

            if(self.include_loglog_plots):
                axs[1].loglog(x, f_l_r_sum[basis_idx, :], label="Sum partial waves")
                
            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.155, right=0.92, hspace=0.2, wspace=0.2)

            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx], x=mid)

            
            axs[0].set_title(f"$U_{basis_idx + 1}^{{(l)}}({xlabel})$")
            axs[0].set_xlabel(f"${xlabel} \\ {x_label_unit}$")
            axs[0].set_ylabel(f"$U_{basis_idx + 1}^{{(l)}}$")
            axs[0].grid(color='lightgray', linestyle='dashed')
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel(f"$\\log {xlabel} \\ {x_label_unit}$")
                axs[1].set_ylabel(f"$\\log V_l({xlabel})$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #f"V_l({xlabel})-partial-wave-sum"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_fits_seperated(self, X_qx_check, ampHandler: AmplitudeHandler, fig_name, base_type, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axs = [ax]

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(X_qx_check)
                for X_idx, X in enumerate(X_qx_check):
                    fit_vals[X_idx] = ampHandler.f_l_fit(basis_idx, l, X)

                axs[0].plot(X_qx_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")

                if(self.include_loglog_plots):
                    axs[1].loglog(X_qx_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                    axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")

            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.155, right=0.92, hspace=0.2, wspace=0.2)

            mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx], x=mid)

            axs[0].set_title(f"$U_{basis_idx + 1}^{{(l)}}(X)$")
            axs[0].set_xlabel("$X$", fontsize="large")
            axs[0].set_ylabel(f"$U_{basis_idx + 1}^{{(l)}}$", fontsize="large")
            axs[0].grid(color='lightgray', linestyle='dashed')
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log X$")
                axs[1].set_ylabel("$\\log V_{l}(X)$")
                axs[1].legend()

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx)  # "fits-seperated-V_l(X)"

            if(self.show_plots):
                plt.show()

            plt.close()

    def plot_pwave_q_amp_fits_seperated(self, q_check, ampHandler: AmplitudeHandler, fig_name, base_type, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(6, 6))

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(q_check)
                for q_idx, q in enumerate(q_check):
                    fit_vals[q_idx] = ampHandler.f_l_q_fit_at(basis_idx, l, q)

                axs[0].plot(q_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")

                if(self.include_loglog_plots):
                    axs[1].loglog(q_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                    axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{pwave_names[l]}-wave", linestyle="-.")
                

            axs[0].set_xlabel("$q$")
            axs[0].set_ylabel("$V_{l}(q)$")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log q$")
                axs[1].set_ylabel("$\\log V_{l}(q)$")
                axs[1].legend()

            #axs[0].set_ylim([-0.001, 0.02])

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx)     #"fits-seperated-V_l(q)"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_FT(self, r_grid, ampHandler: AmplitudeHandler, fig_name, base_type, step_idx: int):
        for basis_idx in range(ampHandler.f_l_q.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(6, 6))

            for l in range(ampHandler.f_l_q.shape[1]):
                
                V_r = np.zeros_like(r_grid)
                for r_idx, r in enumerate(r_grid):
                    V_r[r_idx] = ampHandler.f_l_r_at(basis_idx, l, r)

                axs[0].plot(r_grid, V_r, label=f"{pwave_names[l]}-wave")

                if(self.include_loglog_plots):
                    axs[1].loglog(r_grid, V_r, label=f"{pwave_names[l]}-wave")
                

            axs[0].set_xlabel("$r$")
            axs[0].set_ylabel("$V_{l}(r)$")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log r$")
                axs[1].set_ylabel("$\\log V_{l}(r)$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #"FT-V_l(r)"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_fits(self, X_qx_check, ampHandler: AmplitudeHandler, fig_name, base_type, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(6, 6))

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(X_qx_check)
                for X_idx, X in enumerate(X_qx_check):
                    fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)

                axs[0].plot(X_qx_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")

                if(self.include_loglog_plots):
                    axs[1].loglog(X_qx_check, fit_vals, label=f"{pwave_names[l]}-wave-fit")
                

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel("$V_{l}(X)$")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log X$")
                axs[1].set_ylabel("$\\log V_{l}(X)$")
                axs[1].legend()

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #"fitonly-V_l(X)"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_amp_with_interpolation(self, ampHandler: AmplitudeHandler, f_name: str, base_type, fig_name, step_idx):
        X_check = np.linspace(ampHandler.X[0], ampHandler.X[-1], 1000)

        for basis_idx in range(ampHandler.f_l.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axs = [ax]

            for l in range(ampHandler.f_l.shape[1]):
                axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                axs[0].plot(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"{pwave_names[l]}-wave interp")

                if(self.include_loglog_plots):
                    axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                    axs[1].loglog(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"{pwave_names[l]}-wave interp")
                
            fig.suptitle(self.tensorBasisNamesDict["rho"][basis_idx])

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel(f"${f_name}_l(X)$")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log X$")
                axs[1].set_ylabel(f"$\\log {f_name}_l(X)$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #f"interp-{f_name}_l(X)"

            if(self.show_plots):
                plt.show()

            plt.close()


    def plot_pwave_q_amp_with_interpolation(self, ampHandler: AmplitudeHandler, f_name: str, fig_name, base_type, step_idx):
        q_check = np.linspace(ampHandler.q[0], ampHandler.q[-1], 1000)

        for basis_idx in range(ampHandler.f_l_q.shape[0]):
            if(self.include_loglog_plots):
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(6, 6))

            for l in range(ampHandler.f_l_q.shape[1]):
                axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                axs[0].plot(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"{pwave_names[l]}-wave interp")

                if(self.include_loglog_plots):
                    axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
                    axs[1].loglog(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"{pwave_names[l]}-wave interp")
                

            axs[0].set_xlabel("$q$")
            axs[0].set_ylabel(f"${f_name}_l(q)$")
            axs[0].legend()

            if(self.include_loglog_plots):
                axs[1].set_xlabel("$\\log q$")
                axs[1].set_ylabel(f"$\\log {f_name}_l(q)$")
                axs[1].legend()

            if(self.savefig):
                self.save_active_fig(fig_name, step_idx, base_type, basis_idx) #f"interp-{f_name}_l(q)"

            if(self.show_plots):
                plt.show()

            plt.close()