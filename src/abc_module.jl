using ApproxBayes
# ApproxBayes is not compatible anymore, a workaround is to use `Pkg.add(https://github.com/xiaomingfu2013/ApproxBayes.jl)`


"""
    priors: prior distribution
    err_func: function to compute the error
    data_obs: observed data
"""
function setup_optimization_abc(priors, err_func; target=0.1, nparticles=2000, ϵ1=2e6, MaxFuncEvals, kwargs...)
    sym_priors = create_symbol_dict(priors)
    n_param = length(sym_priors)
    priors = ApproxBayes.Prior(collect(values(sym_priors)))

    function err_func_abc(p, constants, data_obs)
        return err_func(p), nothing
    end

    ABC_kwargs = [
        :nparticles,
        :ϵ1,
        :ϵT,
        :constants,
        :maxiterations,
        :convergence,
        :kernel,
        :α,
    ]
    _kwargs = filter(x -> x[1] in ABC_kwargs, kwargs)

    setup = ABCSMC(err_func_abc, n_param, target, priors; nparticles=nparticles, ϵ1=ϵ1,
        maxiterations=MaxFuncEvals, _kwargs...)
    return setup
end
