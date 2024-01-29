# Simultaneous Identification of Event Change Points and Model Parameters

This repository contains the code for the paper Xiaoming Fu et al. "Simultaneous Identification of Changepoints and Model Parameters in Switching Dynamical Systems" (2024).

## Structure

- `scripts/src` main Julia code used to build the ode models, perform the Bayesian inference. Some of the code that calculates gradient information of the parameters are adapted from the [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) with copyright `Copyright (c) 2016-2020: ChrisRackauckas, Julia Computing.`
- `scripts/plot_scripts` scripts to generate the figures in the paper
- `data` directory to save the generated simulation data and the raw data from [covid-19-data](https://github.com/owid/covid-19-data)