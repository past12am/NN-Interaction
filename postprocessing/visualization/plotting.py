import os

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from data.AmplitudeHandler import AmplitudeHandler


#def plot_form_factor(pd_tau: pd.DataFrame, tensor_basis_elem: str, tensor_basis_elem_idx: int, fig_path=None, save_plot=False):
#    fig = plt.figure(figsize=(10, 9))
#
#    fig.suptitle("Tensor Basis Element " + tensor_basis_elem)
#
#    function_name = "h"
#
#    # Subplot real h
#    ax = fig.add_subplot(2, 2, 1, projection='3d')
#    ax.set_title(f"$\Re({function_name}(X, Z))$")
#    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
#    ax.set_xlabel("$X$")
#    ax.set_ylabel("$Z$")
#    ax.set_zlabel(f"$\Re({function_name})$")
#
#
#    # Subplot imag h
#    ax = fig.add_subplot(2, 2, 2, projection='3d')
#    ax.set_title(f"$\Im({function_name}(X, Z))$")
#    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
#    ax.set_xlabel("$X$")
#    ax.set_ylabel("$Z$")
#    ax.set_zlabel(f"$\Im({function_name})$")
#
#
#
#    function_name = "f"
#
#    # Subplot real f
#    ax = fig.add_subplot(2, 2, 3, projection='3d')
#    ax.set_title(f"$\Re({function_name}(X, Z))$")
#    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
#    ax.set_xlabel("$X$")
#    ax.set_ylabel("$Z$")
#    ax.set_zlabel(f"$\Re({function_name})$")
#
#    # Subplot imag f
#    ax = fig.add_subplot(2, 2, 4, projection='3d')
#    ax.set_title(f"$\Im({function_name}(X, Z))$")
#    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
#    ax.set_xlabel("$X$")
#    ax.set_ylabel("$Z$")
#    ax.set_zlabel(f"$\Im({function_name})$")
#
#
#    if(fig_path is not None and save_plot):
#        plt.savefig(fig_path, dpi=200)
#    plt.show()
#    plt.close()


#def plot_full_amplitude(pd_ff_list, tensor_basis_names, fig_path: str, savefig: bool=False):
#    fig = plt.figure(figsize=(17, 5), constrained_layout=True)
#
#    for basis_idx, pd_ff in enumerate(pd_ff_list):
#        ax = fig.add_subplot(1, 5, basis_idx + 1, projection='3d')
#        ax.set_title(tensor_basis_names[basis_idx])
#        ax.plot_trisurf(pd_ff["X"], pd_ff["Z"], np.real(pd_ff["f"]), cmap=cm.coolwarm)
#        ax.set_xlabel("$X$")
#        ax.set_ylabel("$Z$")
#        ax.set_zlabel(f"$F_{basis_idx}(X, Z)$")
#
#
#    wspace = 0.4   # the amount of width reserved for blank space between subplots
#
#    fig.subplots_adjust(wspace=wspace, top=0.95, bottom=0.05, left=0.05, right=0.95)
#
#    if(savefig):
#        plt.savefig(fig_path, dpi=600)
#    plt.show()
#    plt.close()


#def plot_result(pd_tau, idx_selector, img_path=None, perf_norm=False):
#    fig = plt.figure()
#    ax = plt.axes(projection ='3d')
#    ax.plot_trisurf(pd_tau["X"][idx_selector], pd_tau["Z"][idx_selector], ((1.0/(2 * (2 * math.pi)**4))**2 if perf_norm else 1) * pd_tau["|scattering_amp|2"][idx_selector])
#    ax.set_xlabel("X / 1")
#    ax.set_ylabel("Z / 1")
#    ax.set_zlabel("$|M|^2$")
#    if img_path is not None:
#        plt.savefig(img_path + "/NN-Scattering.png", dpi=400)
#    plt.show()
#    plt.close()



class Plotter:

    def __init__(self, base_path: str, tensorBasisNamesT, tensorBasisNamesTau, tensorBasisNamesRho, savefig) -> None:
        self.base_path = base_path

        self.tensorBasisNamesT = tensorBasisNamesT
        self.tensorBasisNamesTau = tensorBasisNamesTau
        self.tensorBasisNamesRho = tensorBasisNamesRho

        self.savefig = savefig

        # Check if paths exist or create
        if(self.savefig):
            for basis_idx in range(5):
                os.makedirs(self.base_path_for(basis_idx), exist_ok=True) 

            os.makedirs(self.base_path_for(None), exist_ok=True) 


    def base_path_for(self, basis_idx: int=None):
        if (basis_idx is not None):
            return self.base_path + f"/basis-{basis_idx}/"
        else:
            return self.base_path + "/generic/"
    

    def save_active_fig(self, fig_name, step_idx, basis_idx: int=None):
        plt.savefig(self.base_path_for(basis_idx) + "/" + f"{step_idx:02d}__{fig_name}.png")


    def plot_full_amplitude_np(self, X: np.ndarray, Z: np.ndarray, F: np.ndarray, tensor_basis_names, fig_name: str, step_idx: int):
        fig = plt.figure(figsize=(17, 5), constrained_layout=True)

        for basis_idx in range(F.shape[0]):
            ax = fig.add_subplot(1, 5, basis_idx + 1, projection='3d')
            ax.set_title(tensor_basis_names[basis_idx])
            ax.plot_trisurf(X, Z, F[basis_idx, :, :].flatten(), cmap=cm.coolwarm)
            ax.set_xlabel("$X$")
            ax.set_ylabel("$Z$")
            ax.set_zlabel(f"$F_{basis_idx}(X, Z)$")


        wspace = 0.4   # the amount of width reserved for blank space between subplots

        fig.subplots_adjust(wspace=wspace, top=0.95, bottom=0.05, left=0.05, right=0.95)

        if(self.savefig):
            self.save_active_fig(fig_name, step_idx)
        plt.show()
        plt.close()


    def plot_form_factor_np(self, X: np.ndarray, Z: np.ndarray, dressing_f: np.ndarray, dressing_f_name: str, tensor_basis_elem: str, basis_idx: int, fig_name: str, step_idx: int):
        fig = plt.figure(figsize=(10, 5))

        fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

        # Subplot real
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title(f"$\Re({dressing_f_name}(X, Z))$")
        ax.plot_trisurf(X, Z, np.real(dressing_f.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Re({dressing_f_name})$")


        # Subplot imag
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title(f"$\Im({dressing_f_name}(X, Z))$")
        ax.plot_trisurf(X, Z, np.imag(dressing_f.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Im({dressing_f_name})$")


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, basis_idx)
        plt.show()
        plt.close()



    def plot_form_factor_np_side_by_side(self, X1: np.ndarray, Z1: np.ndarray, dressing_f1: np.ndarray, dressing_f_name1: str, xlabel1: str,
                                        X2: np.ndarray, Z2: np.ndarray, dressing_f2: np.ndarray, dressing_f_name2: str, xlabel2: str, 
                                        tensor_basis_elem: str, basis_idx: int, fig_name: str, step_idx: int):
        dressing_f1_params = f"({xlabel1}, Z)"
        dressing_f2_params = f"({xlabel2}, Z)"

        fig = plt.figure(figsize=(10, 9))

        fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

        # Subplot real
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_title(f"$\Re({dressing_f_name1}{dressing_f1_params})$")
        ax.plot_trisurf(X1, Z1, np.real(dressing_f1.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel1}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Re({dressing_f_name1})$")


        # Subplot imag
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_title(f"$\Im({dressing_f_name1}{dressing_f1_params})$")
        ax.plot_trisurf(X1, Z1, np.imag(dressing_f1.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel1}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Im({dressing_f_name1})$")



        # Subplot real
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.set_title(f"$\Re({dressing_f_name2}{dressing_f2_params})$")
        ax.plot_trisurf(X2, Z2, np.real(dressing_f2.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel2}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Re({dressing_f_name2})$")


        # Subplot imag
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.set_title(f"$\Im({dressing_f_name2}{dressing_f2_params})$")
        ax.plot_trisurf(X2, Z2, np.imag(dressing_f2.flatten()), cmap=cm.coolwarm)
        ax.set_xlabel(f"${xlabel2}$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$\Im({dressing_f_name2})$")


        if(self.savefig):
            self.save_active_fig(fig_name, step_idx, basis_idx)
        plt.show()
        plt.close()



    def plotAmplitudes(self, dataloader, step_idx: int):
        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.f[base_idx, :, :], "f", self.tensorBasisNamesTau[base_idx], base_idx, fig_name=f"Amplitude_tau_{base_idx + 1}", step_idx=step_idx)

        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.F[base_idx, :, :], "F", self.tensorBasisNamesT[base_idx], base_idx, fig_name=f"Amplitude_T_{base_idx + 1}", step_idx=step_idx)


    def plotAmplitudes_h(self, dataloader, step_idx: int):
        # TODO note both basis elements in contraction
        for base_idx in range(5):
            self.plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.h[base_idx, :, :], "h", self.tensorBasisNamesTau[base_idx], base_idx, fig_name=f"Amplitude_tauprime_{base_idx + 1}", step_idx=step_idx)


    def plotFullSymAmplitude(self, dataloader_qx, dataloader_dqx, step_idx):
        # Plot Full (Anti-) Symmetric Amplitude
        F_complete = 1/2 * (dataloader_qx.F_flavored - dataloader_dqx.F_flavored)
        self.plot_full_amplitude_np(dataloader_qx.X, dataloader_qx.Z, F_complete, self.tensorBasisNamesT, fig_name="Amplitude_Sym", step_idx=step_idx)


    def plotAmplitudesPartialWaveExpandedAndOriginal(self, grid_var1: np.ndarray, Z: np.ndarray, V_l: np.ndarray, V: np.ndarray, var1_name: str, step_idx: int):
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

            self.plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V[basis_idx, :, :], "V", var1_name,
                                                  X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "V_{check}", var1_name,
                                                  self.tensorBasisNamesRho[basis_idx], basis_idx, fig_name=f"Amplitude_rho_{basis_idx + 1}", step_idx=step_idx)
            

    def plot_pwave_amp_with_fits(self, V_qx_l, X_qx, ampHandler: AmplitudeHandler, step_idx: int):
        for basis_idx in range(V_qx_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(V_qx_l.shape[1]):

                fit_vals = np.zeros_like(X_qx)
                for X_idx, X in enumerate(X_qx):
                    fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)


                axs[0].plot(X_qx, fit_vals, label=f"{l}-wave-fit")
                axs[0].plot(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")

                axs[1].loglog(X_qx, fit_vals, label=f"{l}-wave-fit")
                axs[1].loglog(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")

            fig.suptitle(self.tensorBasisNamesRho[basis_idx])            

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel("$V_{l}(X)$")

            axs[1].set_xlabel("$\log X$")
            axs[1].set_ylabel("$\log V_{l}(X)$")

            axs[0].legend()
            axs[1].legend()

            if(self.savefig):
                self.save_active_fig("fits-V_l(X)", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_amp(self, f_l, x, xlabel, x_label_unit, step_idx: int, max_wave: int=None):
        for basis_idx in range(f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(f_l.shape[1] if max_wave is None else max_wave):
                axs[0].plot(x, f_l[basis_idx, l, :], label=f"{l}-wave")
                axs[1].loglog(x, f_l[basis_idx, l, :], label=f"{l}-wave")
                
            fig.suptitle(self.tensorBasisNamesRho[basis_idx])

            axs[0].set_xlabel(f"${xlabel} \\ {x_label_unit}$")
            axs[0].set_ylabel(f"$V_l({xlabel})$")

            axs[1].set_xlabel(f"$\log {xlabel} \\ {x_label_unit}$")
            axs[1].set_ylabel(f"$\log V_l({xlabel})$")

            axs[0].legend()
            axs[1].legend()

            if(self.savefig):
                self.save_active_fig(f"V_l({xlabel})", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_amp_fits_seperated(self, X_qx_check, ampHandler: AmplitudeHandler, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(X_qx_check)
                for X_idx, X in enumerate(X_qx_check):
                    fit_vals[X_idx] = ampHandler.f_l_fit(basis_idx, l, X)

                axs[0].plot(X_qx_check, fit_vals, label=f"{l}-wave-fit")
                axs[1].loglog(X_qx_check, fit_vals, label=f"{l}-wave-fit")

                axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
                axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
                
            fig.suptitle(self.tensorBasisNamesRho[basis_idx])

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel("$V_{l}(X)$")

            axs[1].set_xlabel("$\log X$")
            axs[1].set_ylabel("$\log V_{l}(X)$")

            axs[0].legend()
            axs[1].legend()

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig("fits-seperated-V_l(X)", step_idx, basis_idx)

            plt.show()
            plt.close()

    def plot_pwave_q_amp_fits_seperated(self, q_check, ampHandler: AmplitudeHandler, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(q_check)
                for q_idx, q in enumerate(q_check):
                    fit_vals[q_idx] = ampHandler.f_l_q_fit_at(basis_idx, l, q)

                axs[0].plot(q_check, fit_vals, label=f"{l}-wave-fit")
                axs[1].loglog(q_check, fit_vals, label=f"{l}-wave-fit")

                axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
                axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
                

            axs[0].set_xlabel("$q$")
            axs[0].set_ylabel("$V_{l}(q)$")

            axs[1].set_xlabel("$\log q$")
            axs[1].set_ylabel("$\log V_{l}(q)$")

            axs[0].legend()
            axs[1].legend()

            #axs[0].set_ylim([-0.001, 0.02])

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig("fits-seperated-V_l(q)", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_amp_FT(self, r_grid, ampHandler: AmplitudeHandler, step_idx: int):
        for basis_idx in range(ampHandler.f_l_q.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l_q.shape[1]):
                
                V_r = np.zeros_like(r_grid)
                for r_idx, r in enumerate(r_grid):
                    V_r[r_idx] = ampHandler.f_l_r_at(basis_idx, l, r)

                axs[0].plot(r_grid, V_r, label=f"{l}-wave")
                axs[1].loglog(r_grid, V_r, label=f"{l}-wave")
                

            axs[0].set_xlabel("$r$")
            axs[0].set_ylabel("$V_{l}(r)$")

            axs[1].set_xlabel("$\log r$")
            axs[1].set_ylabel("$\log V_{l}(r)$")

            axs[0].legend()
            axs[1].legend()

            if(self.savefig):
                self.save_active_fig("FT-V_l(r)", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_amp_fits(self, X_qx_check, ampHandler: AmplitudeHandler, step_idx: int, Ymax: float=None):
        for basis_idx in range(ampHandler.f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l.shape[1]):
                
                fit_vals = np.zeros_like(X_qx_check)
                for X_idx, X in enumerate(X_qx_check):
                    fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)

                axs[0].plot(X_qx_check, fit_vals, label=f"{l}-wave-fit")
                axs[1].loglog(X_qx_check, fit_vals, label=f"{l}-wave-fit")
                

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel("$V_{l}(X)$")

            axs[1].set_xlabel("$\log X$")
            axs[1].set_ylabel("$\log V_{l}(X)$")

            axs[0].legend()
            axs[1].legend()

            if Ymax is not None:
                axs[1].set_ylim(Ymax)

            if(self.savefig):
                self.save_active_fig("fitonly-V_l(X)", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_amp_with_interpolation(self, ampHandler: AmplitudeHandler, f_name: str, step_idx):
        X_check = np.linspace(ampHandler.X[0], ampHandler.X[-1], 1000)

        for basis_idx in range(ampHandler.f_l.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l.shape[1]):
                axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave")
                axs[0].plot(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"{l}-wave interp")

                axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave")
                axs[1].loglog(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"{l}-wave interp")
                
            fig.suptitle(self.tensorBasisNamesRho[basis_idx])

            axs[0].set_xlabel("$X$")
            axs[0].set_ylabel(f"${f_name}_l(X)$")

            axs[1].set_xlabel("$\log X$")
            axs[1].set_ylabel(f"$\log {f_name}_l(X)$")

            axs[0].legend()
            axs[1].legend()

            if(self.savefig):
                self.save_active_fig(f"interp-{f_name}_l(X)", step_idx, basis_idx)

            plt.show()
            plt.close()


    def plot_pwave_q_amp_with_interpolation(self, ampHandler: AmplitudeHandler, f_name: str, step_idx):
        q_check = np.linspace(ampHandler.q[0], ampHandler.q[-1], 1000)

        for basis_idx in range(ampHandler.f_l_q.shape[0]):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))

            for l in range(ampHandler.f_l_q.shape[1]):
                axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave")
                axs[0].plot(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"{l}-wave interp")

                axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave")
                axs[1].loglog(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"{l}-wave interp")
                

            axs[0].set_xlabel("$q$")
            axs[0].set_ylabel(f"${f_name}_l(q)$")

            axs[1].set_xlabel("$\log q$")
            axs[1].set_ylabel(f"$\log {f_name}_l(q)$")

            axs[0].legend()
            axs[1].legend()

            if(self.savefig):
                self.save_active_fig(f"interp-{f_name}_l(q)", step_idx, basis_idx)

            plt.show()
            plt.close()