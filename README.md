# Simultaneous Identification of Changepoints and Model Parameters in Switching Dynamical Systems

This repository contains the code for the paper Xiaoming Fu et al. "Simultaneous Identification of Changepoints and Model Parameters in Switching Dynamical Systems" (2024). To run the code repository locally, please check the `Project.toml` file for the required packages and install them in your local Julia environment.

## Structure

- `scripts/Bayesian_ChangePointStudy` main scripts to run the Bayesian inference for the models in the paper.
- `src` main Julia code used to build the ode models, adjoint sensitivity analysis, and Bayesian inference. Some of the code that calculates gradient information of the parameters are adapted from the [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) with copyright `Copyright (c) 2016-2020: ChrisRackauckas, Julia Computing.`
- `scripts/plot_scripts` scripts to generate the figures in the paper
- `data` directory to save the generated simulation data and the raw data from [covid-19-data](https://github.com/owid/covid-19-data).
- `test` directory to run the tests for the correctness of reverse-mode AD and the gradient calculation of the parameters.
