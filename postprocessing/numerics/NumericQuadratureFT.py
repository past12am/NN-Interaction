import typing
import numpy as np

class NumericQuadratureFT:
    # Via paper method
    # https://www.ams.org/journals/mcom/1956-10-055/S0025-5718-1956-0080994-9/S0025-5718-1956-0080994-9.pdf
    # https://pdf.sciencedirectassets.com/272578/1-s2.0-S0022247X00X04258/1-s2.0-0022247X64900782/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEG8aCXVzLWVhc3QtMSJGMEQCIEUq9ihLdkT7P919IU%2Fpw%2BX9G%2Fkw%2F1b8BMsrv7v3POBPAiB7ODXKDBMjqhmxGPVH3Mt5IZjgCvCYAkyiiA4zSbcqlSq8BQjX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMiAgxDcW%2FWAy2bTumKpAFTBoY7v5FXfbF459T0MjZHQEKe6j%2BAJjiKxkg6pcDZpWhKJfDT2frX%2FfDTfd4%2BXarWi%2Bchnw%2B4DWLH1xXcu1gW21ksRc5wck0jB9VMod59BaypckHPJ8cWgfLnHQO%2Bn5Q83xsoNosmKcdj5PESMeFdKjoIwDxtgw1xGEcISBs58JL6zaHBSkRtXtL5ac15MqrvTfN5zW7OzexXSlNNWjkOkQ4AO5ZD5PO3dKq5%2BC0P1rYQssc1cPAVeU1j4WTJMiv6pGjZGFcp7R%2Fzq7eppmzF8t0edCD2A3OtI4v7m0JXK2NCgxShqXmEKOJEP4mchwpIHmKMrOXfy7LCfxmsunfpJTsa7cuy93EIVKVYc9KNhOXd0K2gk6d4b2or81Fo0m2VD8BlHUBQdujQD2xCmwN2VKMIZWw20gOL%2BIdakUxHiulO3QDfCFN4VKYx%2FNgmpSon2ExlGw1g87K4YsFPC3DA2SHMf2bJ%2Fc%2FjohbnZ909kRUhVRCQb0nO0bQQ43uOJ9Yodhaa5YVyhFqSCJ6hT4nFn%2Bb%2BqxVd1tJrarCdYX6s1y9yIvybHGAEBjJ%2F5%2FY%2BuKT4e7FmCT0%2BIoSBO%2BVEbUmK9g7F85%2BHDwPdkuV7Dyg3BeTESlciKIo5EL9RUJv8A66FK4lV3jnv0sFehFpwOF2P8%2BxanZcYFLy5UjuwONCpBAoD%2FItaU1signq%2BqFJvA03VchtPL1RFJlpdc0cf0THoR7hdFWGR6kpxSx4Fcx2yY97COANMSOZXFjVfgXubBIjC7GQnQVPA9JqgOAAVD7WgYZRRShHxKD7lwioSYthsLflhngtZ003L1MRFqdMKizA81JoBRd%2BKsvKUyMDrvOPvnP3nqlQ3YZV1%2Bx95bFQZAIw9MudsgY6sgE7CeZiCu3qg6MfPYHR8J1g%2FjJ0sTnzHC%2B2fXfr9y9bZqg%2BG56MuPh3N%2BjEOe3DYdntgnB8ms9j2ip4EsUFQw39MC%2Fp1cawdt5HtMFb9SeYeAvkTp8p1P%2BJhRVE%2BzeVEn%2Bxh1nIJ9nJaRW0J4mGiAzXeQ%2BD1FZDb1HV1u6KexE34l%2FnEDy9pg9F0mX%2FgUIGMTpcNsHET94mpQPN4ZZPm9NIwDDEnQzLXZg0D%2FlpnOF6VCr8&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240517T145710Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYU4LHUDMK%2F20240517%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3e5398d475bb298ff982ff2a6e1290998d323ee8835332c90c6eb1a1a817e2fd&hash=4a937c0070e24068e377c2d91e5e6c4e6e618b167f3e4e82ce59679e39b91c5d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0022247X64900782&tid=spdf-0f8efb50-2027-403b-beb6-eaaacae754b0&sid=186d85b29bde93490c5b4db6bff777b08036gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=04135a56565a56565b5b&rr=88547559ac965b07&cc=at


    def __init__(self, N, n_max) -> None:
        self.N = N
        self.n_max = n_max


    def fourierTransform(self, V_q_radial: typing.Callable, r_grid_target: np.ndarray):

        V_r = np.zeros_like(r_grid_target)

        for idx, r in enumerate(r_grid_target):
            integral_kernel_potential = lambda q: q * V_q_radial(q)

            beta_n = self._beta_series_up_to_n(integral_kernel_potential, r) #np.array([2.6191, -1.8693, 2.0741, -1.4098, 1.6853, -1.0791])
            a = self._a_series(beta_n)[0]

            V_r[idx] = - 1/np.square(2 * np.pi) * 1/r * np.pi/r * a

        return V_r


    def _y_j_N(self, j):
        return (2 * j - 1) / (2 * (2 * self.N + 1))
    

    def _W_j_N(self, j):
        i = self.N - j + 1
        return np.square(np.cos(np.pi * self._y_j_N(i))) / (2 * self.N + 1)


    def _S_n(self, n, phi: typing.Callable, x):
        s = 0
        for j in range(1, self.N + 1):
            y_jN = self._y_j_N(j)
            s += self._W_j_N(j) / (np.cos(np.pi * y_jN)) * (phi(np.pi/x * (y_jN + n + 0.5)) + phi(np.pi/x * (-y_jN + n + 0.5)))

        return np.power(-1, n) * s #s if n % 2 == 0 else -s



    def _beta_series_up_to_n(self, phi: typing.Callable, x):
        beta_n = np.zeros(self.n_max + 1)

        beta_n[0] = self._S_n(0, phi, x)
        for n in range(1, self.n_max + 1):
            beta_n[n] = beta_n[n - 1] + self._S_n(n, phi, x)

        return beta_n


    @staticmethod
    def _ai_n(pred_array, n):
        return 0.5 * (pred_array[n] + pred_array[n + 1])

    @staticmethod
    def _a_series(beta_n: np.ndarray):
        max_iter = len(beta_n)

        alpha_in = [np.zeros(i) for i in range(max_iter - 1, 0, -1)]
        alpha_in.insert(0, beta_n)      # beta_n = alpha_0n

        for n in range(max_iter - 1):
            for i in range(len(alpha_in[n]) - 1):
                alpha_in[n + 1][i] = NumericQuadratureFT._ai_n(alpha_in[n], i)

        return alpha_in[-1]