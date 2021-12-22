"""Configuration class for infection estimation simulations."""

# standard
import os
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial

# third party
import numpy as np

# first party
from data_containers import SensorConfig
from deconvolution import deconvolution
from pandas import date_range


@dataclass
class Config:
    JHU_DATA_PATH = '../jhu-csse_confirmed_incidence_prop/'
    DELAY_DISTRIBUTION_DATA_PATH = '../naive_delay_distributions/'
    OUTPUT_DATA_PATH = '../results/'

    @staticmethod
    def get_delay_distribution(as_of,
                               storage_dir = '../data/naive_delay_distributions',
                               prefix = 'delay_distribution_as_of'):
        """Retrieve dictionary with delay distribution values"""
        return pickle.load(
            open(os.path.join(storage_dir, f'{prefix}_{as_of}.p'), 'rb'))

    max_delay_days = 45
    distribution_support = np.arange(1, max_delay_days + 1)
    support_size = len(distribution_support)
    first_data_date = date(2020, 5, 1)
    start_date = date(2020, 10, 1)
    end_date = date(2021, 6, 1)
    ground_truth_date = end_date + timedelta(2 * max_delay_days)

    as_of_range = [d.date() for d in date_range(start_date, end_date)]
    every_10_as_of_range = as_of_range[::10]

    # Top 200 counties by population
    top_counties = {'01073', '01089', '01097', '04013', '04019', '04021',
                    '05119', '06001', '06013', '06019', '06029', '06037',
                    '06053', '06059', '06061', '06065', '06067', '06071',
                    '06073', '06075', '06077', '06081', '06083', '06085',
                    '06095', '06097', '06099', '06107', '06111', '08001',
                    '08005', '08031', '08041', '08059', '09001', '09003',
                    '09009', '10003', '11001', '12009', '12011', '12021',
                    '12031', '12057', '12069', '12071', '12081', '12083',
                    '12086', '12095', '12097', '12099', '12101', '12103',
                    '12105', '12115', '12117', '12127', '13067', '13089',
                    '13121', '13135', '15003', '16001', '17031', '17043',
                    '17089', '17097', '17197', '18003', '18089', '18097',
                    '19153', '20091', '20173', '21111', '22033', '22051',
                    '22071', '24003', '24005', '24031', '24033', '24510',
                    '25005', '25009', '25013', '25017', '25021', '25023',
                    '25025', '25027', '26049', '26081', '26099', '26125',
                    '26161', '26163', '27003', '27037', '27053', '27123',
                    '29095', '29183', '29189', '31055', '32003', '32031',
                    '33011', '34003', '34005', '34007', '34013', '34017',
                    '34021', '34023', '34025', '34027', '34029', '34031',
                    '34039', '35001', '36005', '36029', '36047', '36055',
                    '36059', '36061', '36067', '36071', '36081', '36085',
                    '36103', '36119', '37067', '37081', '37119', '37183',
                    '39017', '39035', '39049', '39061', '39095', '39113',
                    '39151', '39153', '40109', '40143', '41005', '41039',
                    '41051', '41067', '42003', '42011', '42017', '42029',
                    '42045', '42071', '42077', '42091', '42101', '42133',
                    '44007', '45019', '45045', '45079', '47037', '47065',
                    '47093', '47157', '48027', '48029', '48039', '48061',
                    '48085', '48113', '48121', '48141', '48157', '48201',
                    '48215', '48339', '48355', '48439', '48453', '48491',
                    '49035', '49049', '51059', '51107', '51153', '51810',
                    '53011', '53033', '53053', '53061', '53063', '55025',
                    '55079', '55133'}
    megacounties = {'01000', '02000', '04000', '05000', '06000', '08000',
                    '09000', '10000', '11000', '12000', '13000', '15000',
                    '16000', '17000', '18000', '19000', '20000', '21000',
                    '22000', '23000', '24000', '25000', '26000', '27000',
                    '28000', '29000', '30000', '31000', '32000', '33000',
                    '34000', '35000', '36000', '37000', '38000', '39000',
                    '40000', '41000', '42000', '44000', '45000', '46000',
                    '47000', '48000', '49000', '50000', '51000', '53000',
                    '54000', '55000', '56000'}
    states = {'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
              'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me',
              'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm',
              'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
              'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy'}

    state_fips = {'al': '01', 'ak': '02', 'az': '04', 'ar': '05', 'ca': '06',
                  'co': '08', 'ct': '09', 'de': '10', 'dc': '11', 'fl': '12',
                  'ga': '13', 'hi': '15', 'id': '16', 'il': '17', 'in': '18',
                  'ia': '19', 'ks': '20', 'ky': '21', 'la': '22', 'me': '23',
                  'md': '24', 'ma': '25', 'mi': '26', 'mn': '27', 'ms': '28',
                  'mo': '29', 'mt': '30', 'ne': '31', 'nv': '32', 'nh': '33',
                  'nj': '34', 'nm': '35', 'ny': '36', 'nc': '37', 'nd': '38',
                  'oh': '39', 'ok': '40', 'or': '41', 'pa': '42', 'ri': '44',
                  'sc': '45', 'sd': '46', 'tn': '47', 'tx': '48', 'ut': '49',
                  'vt': '50', 'va': '51', 'wa': '53', 'wv': '54', 'wi': '55',
                  'wy': '56'}
    deconv_cv_lambda_grid = np.logspace(1, 3.5, 10)
    deconv_cv_gamma_grid = np.r_[np.logspace(0, 0.2, 6) - 1, [1, 5, 10, 50]]
    deconv_fit_func = partial(deconvolution.deconvolve_tf_cv,
                              k=3,
                              fit_func=deconvolution.deconvolve_tf,
                              lam_cv_grid=deconv_cv_lambda_grid,
                              gam_cv_grid=deconv_cv_gamma_grid)

    ground_truth_indicator = SensorConfig('jhu-csse',
                                          'confirmed_incidence_prop',
                                          'test_truth',
                                          1)
    dv_cli = SensorConfig('doctor-visits', 'smoothed_adj_cli', 'dv', 4)
    fb_cliic = SensorConfig('fb-survey', 'smoothed_hh_cmnty_cli', 'fb', 1)
    chng_cli = SensorConfig('chng', 'smoothed_adj_outpatient_cli', 'chng_cli',
                            4)
    chng_covid = SensorConfig('chng', 'smoothed_adj_outpatient_covid',
                              'chng_covid', 4)
    google_aa = SensorConfig('google-symptoms',
                             'sum_anosmia_ageusia_smoothed_search', 'gs', 1)

    indicator_sensors = {'doctor-visits': dv_cli, 'fb-survey': fb_cliic,
                         'chng-cli': chng_cli, 'chng-covid': chng_covid,
                         'google-symptoms': google_aa}
