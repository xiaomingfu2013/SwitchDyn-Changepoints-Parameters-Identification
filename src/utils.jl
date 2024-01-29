using Random
using Zygote
using DataStructures

include("viz.jl")


function extend_event_time_symbols(priors_tuple::NamedTuple; all_symbols=false)
    symbols_ = []
    for (i, s) in enumerate(keys(priors_tuple.p_var))
        for j in eachindex(priors_tuple.p_var[i])
            push!(symbols_, Symbol(Num(Symbolics.variable(s, join(j)))))
        end
    end
    if haskey(priors_tuple, :p_invar)
        for (i, s) in enumerate(keys(priors_tuple.p_invar))
            if all_symbols
                push!(symbols_, Symbol(s))
            else
                priors_tuple.p_invar[i] isa Distributions.Distribution &&
                    push!(symbols_, Symbol(s))
            end
        end
    end
    for (i, s) in enumerate(keys(priors_tuple.event_times))
        if all_symbols
            push!(symbols_, Symbol(s))
        else
            priors_tuple.event_times[i] isa Distributions.Distribution &&
                push!(symbols_, Symbol(s))
        end
    end
    return Symbol.(symbols_)
end

function create_symbol_dict(priors_tuple::NamedTuple)
    pkeys = extend_event_time_symbols(priors_tuple)
    pdist = vcat(vcat(extract_priors(priors_tuple)...)...)
    @assert length(pkeys) == length(pdist)
    return OrderedDict(zip(pkeys, convert.(Distribution, pdist)))
end

function extract_priors(tuple_priors::NamedTuple)
    dict = []
    for (_, v) in pairs(tuple_priors)
        if v isa NamedTuple
            push!(dict, extract_priors(v))
        elseif typeof(v) <: Distributions.Distribution
            push!(dict, v)
        elseif typeof(v) <: Vector{<:Distributions.Distribution}
            for v_ in v
                push!(dict, v_)
            end
        end
    end
    return dict
end

function sample_tuple(rng::AbstractRNG, tl::NamedTuple)
    dict = OrderedDict()
    for (k, v) in pairs(tl)
        if v isa NamedTuple
            push!(dict, k => sample_tuple(rng, v))
        elseif typeof(v) <: Distributions.Distribution
            push!(dict, k => rand(rng, v))
        elseif typeof(v) <: Vector{<:Distributions.Distribution}
            push!(dict, k => rand.(rng, v))
        else
            push!(dict, k => v)
        end
    end
    return (; dict...)
end

function sample_mean_tuple(tl::NamedTuple)
    dict = OrderedDict()
    for (k, v) in pairs(tl)
        if v isa NamedTuple
            push!(dict, k => sample_mean_tuple(v))
        elseif typeof(v) <: Distributions.Distribution
            push!(dict, k => mean(v))
        elseif typeof(v) <: Vector{<:Distributions.Distribution}
            push!(dict, k => mean.(v))
        else
            push!(dict, k => v)
        end
    end
    return (; dict...)
end

function sample_quantile_tuple(tl::NamedTuple, qt::Float64)
    dict = OrderedDict()
    for (k, v) in pairs(tl)
        if v isa NamedTuple
            push!(dict, k => sample_quantile_tuple(v, qt))
        elseif typeof(v) <: Distributions.Distribution
            push!(dict, k => quantile(v, qt))
        elseif typeof(v) <: Vector{<:Distributions.Distribution}
            push!(dict, k => quantile.(v, qt))
        else
            push!(dict, k => v)
        end
    end
    return (; dict...)
end

function build_ps(param::NamedTuple; return_array=true, scale_func::Function=identity)
    param = PH.value(param)
    p_var =
        reduce(vcat, values(param.p_var)) .*
        scale_func(reduce(vcat, values(param.scaling.SCALING_p_var)))
    # p_var = p_var isa Vector ? p_var : [p_var]
    if !(haskey(param, :p_invar))
        p_invar = eltype(p_var)[]
    else
        p_invar =
            reduce(vcat, values(param.p_invar)) .*
            scale_func(reduce(vcat, values(param.scaling.SCALING_p_invar)))
        p_invar = p_invar isa Vector ? p_invar : [p_invar]
    end
    event_times =
        reduce(vcat, values(param.event_times)) .*
        scale_func(reduce(vcat, values(param.scaling.SCALING_event_times)))
    event_times = event_times isa Vector ? event_times : [event_times]
    if return_array
        return [p_var; p_invar; event_times]
    else
        return (p_var, p_invar, event_times)
    end
end

function unscale_build_ps(param::NamedTuple; return_array=true)
    return build_ps(param; return_array=return_array, scale_func=x -> one(eltype(x)) ./ x)
end

function wrapper_scaled_param_dict(param_::NamedTuple)
    param = PH.value(param_)

    dict = OrderedDict()
    for (i, (k, v)) in enumerate(pairs(param.p_var))
        push!(dict, k => v .* param.scaling.SCALING_p_var[i])
    end
    if haskey(param, :p_invar) && !isempty(param.p_invar)
        for (i, (k, v)) in enumerate(pairs(param.p_invar))
            push!(dict, k => v .* param.scaling.SCALING_p_invar[i])
        end
    end
    for (i, (k, v)) in enumerate(pairs(param.event_times))
        push!(dict, k => v .* param.scaling.SCALING_event_times[i])
    end
    return dict
end


likelihood_lna_dist(mean, var, data) = logpdf(MvNormal(mean, Diagonal(var)), data)

function scale_chain(chn, unflatten, fixed_indices=[]; scale_func=identity)
    chn_ = deepcopy(chn)
    data_chain = chn_.value.data
    var_num = length(chn_.name_map.parameters)
    for row in axes(data_chain, 1), chn_num in axes(data_chain, 3)
        nt = unflatten(data_chain[row, 1:var_num, chn_num])
        ps = build_ps(nt; return_array=true, scale_func=scale_func)
        ps_ = deleteat!(ps, fixed_indices)
        data_chain[row, 1:var_num, chn_num] .= ps_
    end
    return chn_
end

function unscale_chain(chn, unflatten, fixed_indices=[])
    chn_ = deepcopy(chn)
    data_chain = chn_.value.data
    var_num = length(chn_.name_map.parameters)
    for row in axes(data_chain, 1), chn_num in axes(data_chain, 3)
        nt = unflatten(data_chain[row, 1:var_num, chn_num])
        ps = unscale_build_ps(nt; return_array=true)
        ps_ = deleteat!(ps, fixed_indices)
        data_chain[row, 1:var_num, chn_num] .= ps_
    end
    return chn_
end
