using Catalyst
using LinearNoiseApproximation
import LinearNoiseApproximation: find_states_cov_number


include("lotka_volterra_ode_models.jl")
include("stochastic_gene_ode_models.jl")
include("seird_model.jl")
include("solve_event_times.jl")

odemodel_selection(string::String) = getfield(Main, Symbol(string))
function unpack_params(p, num_event::Int, set_of_p_var::Int)
    p_var, p_invar, event_times = p[1:(set_of_p_var * (num_event + 1))],
    p[(set_of_p_var * (num_event + 1) + 1):(end - (num_event))],
    p[(end - (num_event - 1)):end]
    return p_var, p_invar, event_times
end
function unpack_params(p, num_event::Int, set_of_p_var::Int, set_of_event_times::Int)
    p_var, p_invar, event_times = p[1:(set_of_p_var * (num_event + 1))],
    p[(set_of_p_var * (num_event + 1) + 1):(end - (num_event) * set_of_event_times)],
    p[(end - ((num_event) * set_of_event_times - 1)):end]
    return p_var, p_invar, event_times
end

import Base.Cartesian: @nexprs, @ntuple

"""
    split according to the length of the tuple
    Input:
        x: vector
        len_tuple: tuple of length n,
    Example:
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    tuple = (2, 3, 4)
    Return Arguments:
        X = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
"""
function split_vector(x::AbstractVector{T}, len_tuple::Tuple{Vararg{Int}}) where {T}
    return split_vector(x, len_tuple, Val(length(len_tuple)))
end

@generated function split_vector(
    x::AbstractVector{T}, len_tuple::Tuple{Vararg{Int}}, ::Val{N}
) where {T,N}
    quote
        len = 0
        @nexprs $N (n) -> (s_n = len + 1; e_n = len + len_tuple[n]; len = e_n)
        X = @ntuple $N (n) -> (reshape(view(x, s_n:e_n), len_tuple[n]))
        return X
    end
end

function find_event_time(t, event_times, set_of_event_times::Int)
    if set_of_event_times == 1
        return searchsortedfirst(event_times, t)
    end
    num_event_times = length(event_times) ÷ set_of_event_times
    event_time_idxs = Vector{Int}(undef, set_of_event_times)
    for i in 1:set_of_event_times
        event_time_idxs[i] = searchsortedfirst(
            event_times[((i - 1) * num_event_times + 1):(i * num_event_times)], t
        )
    end
    return event_time_idxs
end

"""
    find_event_time(t, event_times, num_event_times_list)
    event_times = [τ_beta_1, τ_beta_2, τ_beta_3, τ_sigma_1, τ_sigma_2]
    num_event_times_list = [3, 2] means that there are 3 beta events and 2 sigma events
"""
function find_event_time(t, event_times, len_split_event_times::Tuple{Vararg{Int}})
    set_of_event_times = length(len_split_event_times)
    if set_of_event_times == 1
        return searchsortedfirst(event_times, t)
    end
    event_time_idxs = Vector{Int}(undef, set_of_event_times)
    event_times_split = split_vector(event_times, len_split_event_times)
    for i in eachindex(len_split_event_times)
        event_time_idxs[i] = searchsortedfirst(event_times_split[i], t)
    end
    return event_time_idxs
end

function compute_len_split_ps(orig_pt::NamedTuple)
    return length.(build_ps(orig_pt; return_array=false))
end

function compute_len_split_event_times(orig_pt::NamedTuple)
    return length.(values(orig_pt.p_var)) .- 1
end
