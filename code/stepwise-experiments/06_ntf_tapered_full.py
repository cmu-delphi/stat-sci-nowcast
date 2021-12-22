# standard
import json
import logging
import os
import pickle
import time
from datetime import timedelta
from functools import partial
from multiprocessing import Pool

# third party
import numpy as np
import pandas as pd

# first party
from config import Config
from data_containers import LocationSeries
from deconvolution import deconvolution


def run_deconvolution(as_of, kernel_dict, window_length, fit_func,
                      convolved_response_prefix, convolved_truth_indicator,
                      overwrite = False):
    """Perform a single run of deconvolution."""
    logging.info(f'Starting {as_of}')
    end_date = as_of - timedelta(convolved_truth_indicator.lag)

    if not overwrite and os.path.isfile(
            os.path.join(Config.OUTPUT_DATA_PATH, f'ntf_tapered/as_of'
                                                  f'_{as_of}.p')):
        logging.info(f'{as_of} exists, skipping')
        return True

    convolved_ground_truth = pickle.load(
        open(convolved_response_prefix + f'_{as_of}.p', 'rb'))

    start_date = as_of - timedelta(window_length)
    start_date = max(start_date, Config.first_data_date)
    full_dates = pd.date_range(start_date, end_date)

    out = {}
    for i, (loc, data) in enumerate(convolved_ground_truth.items()):
        try:
            start_time = time.time()
            metadata_file = os.path.join(Config.OUTPUT_DATA_PATH,
                                         f'ntf_tapered/metadata/ntf_tapered'
                                         f'_{data.geo_value}_{as_of}.data')
            signal = data.get_data_range(start_date, end_date, 'locf')
            est = fit_func(y=np.array(signal), x=np.arange(1, len(signal) + 1),
                           kernel_dict=kernel_dict, as_of_date=as_of,
                           location=data.geo_value, output_tuning=True,
                           output_tuning_file=metadata_file)

            # We only store the first n-1 estimates because the
            # reporting delay distribution is not supported on 0.
            out[data.geo_value] = LocationSeries(data.geo_value, data.geo_type,
                                                 dict(zip(full_dates[:-1],
                                                          est[:-1])))

            # Add additional metadata
            metadata = json.load(open(metadata_file, 'r'))
            metadata['method'] = 'ntf_tapered'
            metadata['deconvolution_window'] = len(full_dates)
            metadata['geo_type'] = data.geo_type
            metadata['has_extrapolate'] = False
            metadata['data_lag'] = convolved_truth_indicator.lag
            metadata['runtime'] = time.time() - start_time
            json.dump(metadata, open(metadata_file, 'w'))
        except Exception as e:
            logging.warning(f'Failed {loc}\n{e}')
            continue

    pickle.dump(out, open(
        os.path.join(Config.OUTPUT_DATA_PATH, f'ntf_tapered/as_of_{as_of}.p'),
        'wb'))
    logging.info(f'Finished {as_of}')


if __name__ == '__main__':
    overwrite = False
    as_of_date_range = Config.as_of_range
    convolved_truth_indicator = Config.ground_truth_indicator

    logging.basicConfig(level=logging.DEBUG,
                        filename=f'logs/ntf_tapered_full_05.log', filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    ntf_tapered = partial(
        deconvolution.deconvolve_tf_cv, k=3,
        fit_func=partial(deconvolution.deconvolve_tf, natural=True),
        lam_cv_grid=np.r_[np.logspace(1, 3.5, 10), [5000, 8000, 15000]],
        gam_cv_grid=np.r_[np.logspace(0, 0.2, 6) - 1, [1, 5, 10, 50]],
        verbose=False)

    kernel_len = Config.max_delay_days
    window_length = 2 * kernel_len
    convolved_response_prefix = Config.JHU_DATA_PATH + f'{convolved_truth_indicator.source}_{convolved_truth_indicator.signal}'

    pool = Pool(4)
    pool_results = []
    for as_of in as_of_date_range:
        kernel_dict = Config.get_delay_distribution(as_of,
                                                    storage_dir='../data/km_delay_distributions/')
        pool_results.append(pool.apply_async(run_deconvolution, args=(
            as_of, kernel_dict, window_length, ntf_tapered,
            convolved_response_prefix, convolved_truth_indicator, overwrite,)))

    pool_results = [proc.get() for proc in pool_results]
