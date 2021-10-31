# Experiments with tensorflow_probability

Primarily for Bayesian statistics motivated by the [available tutorials](https://www.tensorflow.org/probability/overview) and Gelman's "Bayesian Data Analysis". The notebooks, not in any particular order are:

* [bayesian_switchpoint.ipynb](https://github.com/vasasav/tensorflow_probability_experiments/blob/main/bayesian_switchpoint.ipynb) follows the online tutorial at the start, but then introduces some simplifications to arrive at a similar result. The aim is to detect point at which the rate of disasters changes

* [linear_regression.ipynb](https://github.com/vasasav/tensorflow_probability_experiments/blob/main/linear_regression.ipynb) Runs a simple linear regression on generated data. The aim is to examine how reliably one can recover the key parameters of regression, such as intercept and gradient, using Bayesian statistics and sampled data points.

* [mcmc_experiments.ipynb](https://github.com/vasasav/tensorflow_probability_experiments/blob/main/mcmc_experiments.ipynb) is aimed at closer inspection of the MCMC mechanism in TFP. In particular, I generate mulitmodal 2D PDF and study how MCMC locates the main mass of that PDF and samples from it.

* [normal_model.ipynb](https://github.com/vasasav/tensorflow_probability_experiments/blob/main/normal_model.ipynb). Recovery of mean and variance from the sampled points of a normally distributed random variable.

* [normal_model_variance_mapped.ipynb](https://github.com/vasasav/tensorflow_probability_experiments/blob/main/normal_model_variance_mapped.ipynb). Recovery of mean and variance from the sampled points of a normally distributed random variable. Here the variance is explored via logarithm, which means that the variables explored by the MCMC chain (mean and log of variance) are both well-defined on the entirety of the real-line.
