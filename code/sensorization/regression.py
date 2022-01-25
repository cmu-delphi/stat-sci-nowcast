# standard
import logging
from datetime import date, timedelta

# third party
import numpy as np

# first party
from data_containers import LocationSeries
from pandas import date_range


def compute_regression_sensor(day: date,
                              covariate: LocationSeries,
                              response: LocationSeries,
                              weights: np.ndarray,
                              include_intercept: bool = True,
                              use_weights: bool = True):
    """External function to compute regression sensors.

        Args:
            day: The date to produce the regression prediction.
            covariate: The input LocationSeries to fit model over.
            response:  The input LocationSeries to fit model on.
            weights: The weights applied on the right-boundary.
            include_intercept: A boolean to include an intercept in the model.
            use_weights: A boolean to perform weighted regression.

        Returns:
            LocationSeries with the fitted values and prediction.
        """
    try:
        first_day = max(min(covariate.dates), min(response.dates))
        train_Y = response.get_data_range(first_day, day)
        train_covariates = covariate.get_data_range(first_day, day)
        train_dates = np.array([d.date() for d in date_range(first_day, day)])

        # If covariates are observed at a time past the response, we produce
        # out-of-sample estimates for these times
        max_covariate_date = max(covariate.dates)
        test_covariates, test_dates = None, None
        if max_covariate_date > day:
            test_covariates = covariate.get_data_range(day + timedelta(1),
                                                       max_covariate_date)
            test_dates = np.array(
                [d.date() for d in date_range(day + timedelta(1),
                                              max_covariate_date)])
    except ValueError:
        raise
        return None
    if not train_Y:
        return None

    non_nan_idx = [i for i, a in enumerate(zip(train_Y, train_covariates)) if
                   not (np.isnan(a[0]) or np.isnan(a[1]))]
    if not non_nan_idx:
        return None
    train_dates = train_dates[np.array(non_nan_idx)]
    non_nan_values = [(i, j) for i, j in zip(train_Y, train_covariates) if
                      not (np.isnan(i) or np.isnan(j))]
    train_Y, train_covariates = zip(*non_nan_values) if non_nan_values else (
        [], [])
    if len(train_Y) < 5:
        logging.warning("Insufficient observations")
        return None

    n = len(train_covariates)
    train_Y = np.array(train_Y)
    train_covariates = np.array(train_covariates)
    X = np.ones((n, 1 + include_intercept))
    X[:, -1] = train_covariates

    if use_weights:
        m = weights.size
        W = np.zeros((n,))
        # ensure max weight is 1
        cumsum_weights = np.cumsum(weights)[-1]
        W[-m:] = (1 - (np.cumsum(weights) / cumsum_weights))[::-1]
        wX = np.diag(np.sqrt(W)) @ X
        wtrain_Y = np.multiply(np.sqrt(W), train_Y)
        B = np.linalg.inv(wX.T @ wX) @ wX.T @ wtrain_Y
    else:
        B = np.linalg.inv(X.T @ X) @ X.T @ train_Y

    Yhat = (X @ B).flatten()
    dates = train_dates

    if test_covariates is not None and test_dates is not None:
        test_n = len(test_covariates)
        test_X = np.ones((test_n, 1 + include_intercept))
        test_X[:, -1] = np.array(test_covariates)
        Yhat_test = test_X @ B
        Yhat = np.concatenate((Yhat, Yhat_test.flatten()))
        dates = np.concatenate((dates, test_dates))

    return LocationSeries(response.geo_value, response.geo_type,
                          dict(zip(dates, Yhat)))
