# standard
import json
from ctypes import *
from datetime import date, timedelta
from functools import partial
from typing import Callable, Dict

# third party
import numpy as np
# first party
from scipy.linalg import toeplitz
from scipy.sparse import diags as band

# load DP algorithm
so_file = "deconvolution/dp_1d_c.so"
c_dp_1d = CDLL(so_file)


def _construct_convolution_matrix(signal: np.ndarray, kernel: np.ndarray,
                                  norm: bool = False) -> np.ndarray:
    """
    Constructs full convolution matrix (n+m-1) x n,
    where n is the signal length and m the kernel length.

    Parameters
    ----------
    signal
        array of values to convolve
    kernel
        array with convolution kernel values
    norm
        boolean whether to normalize rows to sum to sum(kernel)

    Returns
    -------
        convolution matrix
    """
    n = signal.shape[0]
    m = kernel.shape[0]
    padding = np.zeros(n - 1)
    first_col = np.r_[kernel, padding]
    first_row = np.r_[kernel[0], padding]

    P = toeplitz(first_col, first_row)

    if norm:
        scale = P.sum(axis=1) / kernel.sum()
        return P / scale[:, np.newaxis]
    return P


def _construct_day_specific_convolution_matrix(y, run_date, delay_kernels):
    """Construct convolution matrix with time-varying rows."""
    n = y.shape[0]
    if n > len(delay_kernels):
        assert ValueError('Not enough dates in delay_kernel dictionary')

    kernel_length = len(delay_kernels[run_date])
    C = np.zeros((n, n))
    first_kernel_date = min(delay_kernels.keys())
    for i in range(C.shape[0]):
        kernel_day = max(run_date - timedelta(i), first_kernel_date)
        end_index = max(0, C.shape[1] - i)
        start_index = max(0, end_index - kernel_length)
        day_specific_kernel = np.array(delay_kernels[kernel_day][::-1])
        row = C.shape[0] - i - 1
        if end_index > 0:
            C[row, start_index:end_index] = day_specific_kernel[
                                            -(end_index - start_index):]
    return C, np.array(delay_kernels[run_date])


def _construct_poly_interp_mat_right_only(x, n, k):
    assert k == 3, "poly interpolation matrix only constructed for k=3"
    S = np.zeros((n, n - k + 1))
    S[n - 2, n - 4] = (x[n - 3] - x[n - 2]) / (x[n - 3] - x[n - 4])
    S[n - 2, n - 3] = (x[n - 2] - x[n - 4]) / (x[n - 3] - x[n - 4])
    S[n - 1, n - 4] = (x[n - 3] - x[n - 1]) / (x[n - 3] - x[n - 4])
    S[n - 1, n - 3] = (x[n - 1] - x[n - 4]) / (x[n - 3] - x[n - 4])
    S[:(n - 2), :] = np.eye(n - k + 1)
    return S


def _impute_with_neighbors(x: np.ndarray) -> np.ndarray:
    """
    Impute missing values with the average of the elements immediately
    before and after.

    Parameters
    ----------
    x
        signal with missing values

    Returns
    -------
        imputed signal
    """
    # handle edges
    if np.isnan(x[0]):
        x[0] = x[1]

    if np.isnan(x[-1]):
        x[-1] = x[-2]

    imputed_x = np.copy(x)
    for i, (a, b, c) in enumerate(zip(x, x[1:], x[2:])):
        if np.isnan(b):
            imputed_x[i + 1] = (a + c) / 2

    assert np.isnan(imputed_x).sum() == 0

    return imputed_x


def deconvolve_tf(y: np.ndarray, x: np.ndarray,
                  C: np.ndarray,
                  kernel: np.ndarray, lam: float, gam: float,
                  n_iters: int = 200, k: int = 3,
                  natural: bool = True,
                  clip: bool = False) -> np.ndarray:
    """
        Perform deconvolution with trend filtering smoothing and tapered
        smoothing.

        Parameters
        ----------
            y
               array of values to convolve
            x
               array of positions
            C
               convolution matrix operator
            kernel
               array with convolution kernel values
            lam
               trend filtering parameter
            gam
               tapered smoothing parameter
            n_iters
               number of ADMM interations to perform
            k
               order of the trend filtering penalty
            natural
               Boolean to apply natural constraints on the right boundary
            clip
               Boolean to clip count values to [0, infty)

        Returns
        -------
           array of the deconvolved signal values
    """

    assert k == 3, "Natural TF only implemented for k=3"
    n = y.shape[0]
    m = kernel.shape[0]
    C = C[:n]
    rho = lam  # set equal
    D = band([-1, 1], [0, 1], shape=(n - 1, n)).toarray()
    D = np.diff(D, n=k, axis=0)
    P = _construct_poly_interp_mat_right_only(x, n, k)
    D_m = band([-1, 1], [0, 1], shape=(n - 1, n)).toarray()
    D_m[:-m, :] = 0

    # kernel weights for tapered smoothing
    weights = np.ones((D_m.shape[0],))
    weights[-m:] = np.cumsum(kernel[::-1])
    weights /= np.max(weights)
    Gam = gam * np.diag(weights)

    # polynomial interpolation
    if natural:
        C = C @ P
        D = D @ P
        D_m = D_m @ P

    # pre-calculations
    Cty = C.T @ y
    first_x_update = np.linalg.inv(
        (2 * D_m.T @ Gam @ D_m) + C.T @ C + rho * D.T @ D)
    alpha_k = np.zeros(n - k - 1)
    u_k = np.zeros(n - k - 1)
    x_k = first_x_update @ (Cty + rho * D.T @ (alpha_k + u_k))
    for i in range(n_iters):
        x_k = first_x_update @ (Cty + rho * D.T @ (alpha_k + u_k))

        c_dp_1d.read.argtypes = c_int, POINTER(c_double), c_double, POINTER(
            c_double)
        c_dp_1d.read.restype = None
        x = D @ x_k
        alpha_k = (c_double * len(x))()
        x_c = (c_double * len(x))(*x)
        c_dp_1d.tf_dp(c_int(len(x)), x_c, c_double(lam / rho), alpha_k)
        u_k = u_k + alpha_k - D @ x_k

    if natural:
        x_k = P @ x_k
    if clip:
        x_k = np.clip(x_k, 0, np.infty)
    return x_k


def deconvolve_tf_cv(y: np.ndarray, x: np.ndarray,
                     kernel_dict: Dict,
                     as_of_date: date,
                     fit_func: Callable = deconvolve_tf,
                     lam_cv_grid: np.ndarray = np.logspace(1, 3.5,
                                                           10),
                     gam_cv_grid: np.ndarray = np.r_[
                         np.logspace(0, 0.2, 6) - 1, [1, 5, 10,
                                                      50]],
                     gam_n_folds: int = 7, n_iters: int = 200,
                     k: int = 3,
                     clip: bool = False, verbose: bool = False,
                     output_tuning = False,
                     output_tuning_file = None,
                     location = "") -> np.ndarray:
    """
       Run cross-validation to tune smoothness over deconvolve_tf.
       First, leave-every-third-out CV is performed over lambda, fixing gamma=0.
       After choosing the lambda with the smallest squared error, forward
       validation is done to select gamma.

       Parameters
       ----------
       y
           array of values to convolve
       x
           array of positions
       kernel_dict
           dictionary of convolution kernel values, indexed by as_of dates
       fit_func
           deconvolution function to use
       lam_cv_grid
           grid of trend filtering penalty values to search over
       gam_cv_grid
           grid of second boundary smoothness penalty values to search over
       gam_n_folds
           number of splits for forward cv (see above documentation)
       n_iters
           number of ADMM interations to perform.
       k
           order of the trend filtering penalty.
       clip
           Boolean to clip count values to [0, infty)
       verbose
           Boolean whether to print debug statements
       output_tuning
           Boolean whether to output a file storing metadata on tuning
           parameters
       output_tuning_file
           name of output file, not used if None or output_tuning is False
       location
           string with identifying geographic location, for storage purposes

       Returns
       -------
           array of the deconvolved signal values
    """
    fit_func = partial(fit_func, n_iters=n_iters, k=k, clip=clip)
    n = y.shape[0]
    lam_cv_loss = np.zeros((lam_cv_grid.shape[0],))
    gam_cv_loss = np.zeros((gam_cv_grid.shape[0],))

    def _linear_extrapolate(x0, y0, x1, y1, x_new):
        return y0 + ((x_new - x0) / (x1 - x0)) * (y1 - y0)

    # use leave-every-third-out cv for finding lambda, this controls smoothness
    # of entire curve
    for i in range(3):
        test_split = np.zeros((n,), dtype=bool)
        test_split[i::3] = True
        for j, reg_par in enumerate(lam_cv_grid):
            x_hat = np.full((n,), np.nan)
            C, kernel = _construct_day_specific_convolution_matrix(
                y[~test_split],
                as_of_date,
                kernel_dict)
            x_hat[~test_split] = fit_func(y=y[~test_split], x=x[~test_split],
                                          lam=reg_par,
                                          gam=0, C=C, kernel=kernel)
            x_hat = _impute_with_neighbors(x_hat)
            C, _ = _construct_day_specific_convolution_matrix(x_hat, as_of_date,
                                                              kernel_dict)
            y_hat = (C @ x_hat)[:len(x_hat)]
            lam_cv_loss[j] += np.sum((y[test_split] - y_hat[test_split]) ** 2)

    lam = lam_cv_grid[np.argmin(lam_cv_loss)]

    # use forward cv to find gamma, this controls smoothness of
    # right-boundary curve
    for i in range(1, gam_n_folds + 1):
        for j, reg_par in enumerate(gam_cv_grid):
            x_hat = np.full((n - i + 1,), np.nan)
            C, kernel = _construct_day_specific_convolution_matrix(y[:(n - i)],
                                                                   as_of_date,
                                                                   kernel_dict)
            x_hat[:(n - i)] = fit_func(y=y[:(n - i)], x=x[:(n - i)],
                                       gam=reg_par, lam=lam,
                                       C=C, kernel=kernel)
            pos = x[:(n - i + 1)]
            x_hat[-1] = _linear_extrapolate(pos[-3], x_hat[-3], pos[-2],
                                            x_hat[-2],
                                            pos[-1])
            C, _ = _construct_day_specific_convolution_matrix(x_hat, as_of_date,
                                                              kernel_dict)
            y_hat = (C @ x_hat)[:len(x_hat)]
            gam_cv_loss[j] += np.sum((y[:(n - i + 1)][-1:] - y_hat[-1:]) ** 2)

    gam = gam_cv_grid[np.argmin(gam_cv_loss)]
    if verbose: print(f"Chosen parameters: lam:{lam:.4}, gam:{gam:.4}")
    C, kernel = _construct_day_specific_convolution_matrix(y, as_of_date,
                                                           kernel_dict)
    x_hat = fit_func(y=y, x=x, lam=lam, gam=gam, C=C, kernel=kernel)

    if output_tuning:
        tuning_dict = {'geo_value': location, 'as_of': str(as_of_date),
                       'n_iters': n_iters, 'clip': clip, 'k': k,
                       'lam_grid': lam_cv_grid.tolist(),
                       'gam_grid': gam_cv_grid.tolist(),
                       'lam_loss': lam_cv_loss.tolist(),
                       'gam_loss': gam_cv_loss.tolist(), 'best_lam': float(lam),
                       'best_gam': float(gam)}
        json.dump(tuning_dict, open(output_tuning_file, 'w'))

    return x_hat
