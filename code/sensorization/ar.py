# standard
from datetime import date, timedelta
from typing import List

# third party
import numpy as np
from pandas import date_range

# first party
from data_containers import LocationSeries

def _running_lag_matrix(arr: np.ndarray, window: int):
    """Create lag matrix where each column is a lag of the input array.

    Args:
        arr: The input array to lag.
        window: the number of lags to construct.

    Returns:
        The lagged matrix.
    """
    shape = list(arr.shape)
    shape[-1] -= (window - 1)
    assert (shape[-1] > 0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[-1],))


def _standardize(data: np.ndarray):
    """Standardize the columns of the input matrix.

    Args:
        data: The data to standardize.

    Returns:
        A tuple of the standardized data and the means and standard deviations of each
        column.
    """
    means = np.mean(data, axis=0)
    stddevs = np.std(data, axis=0, ddof=1)
    data = (data - means) / stddevs
    return data, means, stddevs


def _ar_fit_weighted(values: np.ndarray,
                     truth_values: np.ndarray,
                     weights: np.ndarray,
                     lags: List[int],
                     lambda_: float):
    """Fit weighted AR model.

    Args:
        values: The training values for the AR model.
        truth_values: The ground truth values for the AR model.
        weights: The weights applied on the right-boundary.
        lags: The list of AR lags to use.
        lambda_: The ridge regularization parameter.

    Returns:
        The vector of fitted coefficients, the training matrix X, and the standardized
        means and standard deviations.
    """
    max_lag = max(lags)
    num_observations = len(values) - max_lag
    if num_observations < 2 * (max_lag + 1):  # 1 for intercept
        return None, None, None
    X = _running_lag_matrix(values[:-1], max_lag)
    X = X[:, [l - 1 for l in lags]]  # subtract 1 for index by 0
    X, means, stddevs = _standardize(X)
    Y = truth_values[max_lag:, None]
    lambda_ *= num_observations  # scale with increasing length

    # decaying weights
    m = weights.size
    W = np.zeros((X.shape[0],))

    # ensure max weight is 1
    cumsum_weights = np.cumsum(weights)[-1]
    W[-m:] = (1 - (np.cumsum(weights) / cumsum_weights))[::-1]

    # augmentation
    wX_int = np.sqrt(np.diag(W)) @ np.hstack((np.ones((X.shape[0], 1)), X))
    aug_X = np.diag(
        np.concatenate(([0],  # intercept is not penalized
                        np.sqrt(lambda_) * np.ones(len(lags)))))
    wX_int = np.vstack((wX_int, aug_X))
    wY = np.multiply(np.sqrt(W), Y.reshape(-1, )).reshape(-1, 1)
    wY_int = np.vstack((wY, np.zeros((len(lags) + 1, 1))))

    B = np.linalg.inv(wX_int.T @ wX_int) @ wX_int.T @ wY_int
    return B, X, means, stddevs


def compute_ar_sensor(day: date,
                      values: LocationSeries,
                      truth_values: LocationSeries,
                      weights: np.ndarray,
                      lags: List[int],
                      lambda_: float):
    """External function to compute the AR sensor.

    Args:
        day: The date to produce AR prediction.
        values: The input LocationSeries to fit model over (covariate).
        truth_values:  The input LocationSeries to fit model on (response).
        weights: The weights applied on the right-boundary.
        lags: The list of AR lags to use.
        lambda_: The ridge regularization parameter.

    Returns:
        LocationSeries with the fitted values and prediction.
    """
    previous_day = day - timedelta(1)
    try:
        min_date = max(min(values.dates), min(truth_values.dates))
        window = values.get_data_range(min_date, previous_day)
        truth_window = truth_values.get_data_range(min_date, previous_day)
        dates = [d.date() for d in date_range(min_date, day)]
        dates = dates[max(lags):]
    except ValueError:
        return None

    # Fit weighted model.
    B, X, means, stddevs = _ar_fit_weighted(np.array(window),
                                            np.array(truth_window),
                                            np.array(weights),
                                            lags, lambda_)
    if B is None:
        return None

    # Obtain predictions.
    new_X = np.hstack((1, (np.array(window)[-np.array(lags[::-1])] - means) /
                       stddevs))
    Yhat = (np.hstack((np.ones((X.shape[0], 1)), X)) @ B).flatten()
    Yhat = np.concatenate((Yhat, new_X @ B))
    if len(dates) != Yhat.size:
        raise ValueError

    return LocationSeries(truth_values.geo_value, truth_values.geo_type,
                          dict(zip(dates, Yhat)))
