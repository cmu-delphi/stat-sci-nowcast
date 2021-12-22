# Real-Time Estimation of COVID-19 Infections via Deconvolution and Sensor Fusion

This directory contains supplementary code for "Real-Time Estimation of COVID
-19 Infections via Deconvolution and Sensor Fusion".

## Description

Script files with `.py` extensions are typically used to run an experiment
, and `.ipynb` files are used to evaluate results. Several experiments have
 two files: a `.py` and `.ipynb` file pair, where the analysis depends on
  both files. The order of the experiments are ordered numerically:

* `00_generate_naive_delay_dist.ipynb` creates the estimated delay
 distributions;
* `01_generate_ground_truth.ipynb` estimates the "ground truth" infection
 curve using trend filtering (TF), after enough time has passed to remove the
  right-truncation bias;
* `02_natural_contraints.py` and `02_natural_contraints.ipynb` compare the
 retrospective TF estimates compared to those from TF with a natural
  constraint on the right-boundary (NTF);
* `03_tapered_smoothing.py` and `03_tapered_smoothing.ipynb` compare NTF to
 NTF with an additional tapered smoothing penalty;
* `04_generate_KM_delay_dist.ipynb` creates the KM-adjusted delay distributions;
* `05_deconvolution_window.py` and `05_deconvolution_window.ipynb` compare
 the performance of NTF (tapered) using various deconvolution windows of 2d
 , 4d, and all past, where d is delay distribution support size;
* `06_ntf_tapered_full.py` runs NTF (tapered) with a 2d deconvolution window
  over the entire evaluation period;
* `07_generate_sensors.py` fits the sensor models on the indicator signals
, as well as the autoregressive sensor;
* `08_sensor_fusion.ipynb` runs the fusion methods on the generated sensors.
 
There are also several miscellaneous scripts:
* `evaluate_delay_distributions.ipynb` which creates the figures comparing
 the naive and KM-adjusted delay distributions;
* `plot_boundary_regularization.ipynb` which creates the figures showing the
 affect of additional regularization on the right boundary;
* `evaluate_fusion_methods.ipynb` which creates the figures analyzing the
 performance of the sensors and sensor fusion methods over the evaluation
  period.

Utilities needed to run experiments are also available:
* `epidata.py` is used to retrieve the data used in the experiments from the
 [COVIDcast API](https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html).
* `data_containers.py` holds custom classes for managing data and other
 variables.
* `config.py` holds various configuration variables used throughout the
 experiments.
* `fusion/`  contains functions to ensemble the sensors through sensor fusion.
* `sensorization/` contains functions for fitting the linear regression
 (`regression.py`) and autoregressive models (`ar.py`) used in generating
  sensors.
* `deconvolution/` contains functions for optimizing the deconvolution
 objective, which are primarily located in `deconvolution.py`. `dp_1d.c` and
  `dp_1d.h` are helper functions that it calls (see Prerequisites section for
   more information).
  

## Prerequisites

The function to solve the 1-dimensional fused lasso problem is given in
 `deconvolution/dp_1d.c`, originally written by the Arnold et al. (credits in
  the file). To use it, it needs to be compiled to a `.so` file, for instance
  , through the command:
```
> cc -fPIC -shared -o dp_1d_c.so dp_1d.c
```

The resulting file `dp_1d_c.so` is called in `deconvolution/deconvolution.py`.
