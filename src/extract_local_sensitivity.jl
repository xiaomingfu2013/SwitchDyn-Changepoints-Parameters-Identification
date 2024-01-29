
struct ForwardDiffSensitivity{CS, CTS}
    du0_sense::Bool # performance improvement if the sensitivity of the initial condition is not needed
end

function ForwardDiffSensitivity(; du0_sense=false, chunk_size = 0, convert_tspan = nothing)
    ForwardDiffSensitivity{chunk_size, convert_tspan}(du0_sense)
end

has_original_jac(S) = isdefined(S, :original_jac) && S.original_jac !== nothing

struct ODEForwardSensitivityProblem{iip, A}
    sensealg::A
end

function ODEForwardSensitivityProblem(f::F, args...; kwargs...) where {F}
    ODEForwardSensitivityProblem(ODEFunction(f), args...; kwargs...)
end

function ODEForwardSensitivityProblem(prob::ODEProblem, alg; kwargs...)
    ODEForwardSensitivityProblem(prob.f, prob.u0, prob.tspan, prob.p, alg; kwargs...)
end



function seed_duals(x::AbstractArray{V}, f,
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x, typemax(Int64))) where {V,
    N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, V})
    duals = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, eltype(vec(x))))}.(vec(x), seeds)
end

function seed_duals(x::Number, f,
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x, typemax(Int64))) where {N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, typeof(x)})
    duals = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, typeof(x)))}(x, seeds[1])
end

has_continuous_callback(cb::DiscreteCallback) = false
has_continuous_callback(cb::ContinuousCallback) = true
has_continuous_callback(cb::CallbackSet) = !isempty(cb.continuous_callbacks)

function ODEForwardSensitivityProblem(f::DiffEqBase.AbstractODEFunction, u0,
    tspan, p, alg::ForwardDiffSensitivity;
    du0 = zeros(eltype(u0), length(u0), length(p)), # perturbations of initial condition
    dp = I(length(p)), # perturbations of parameters
    kwargs...)
    num_sen_par = size(du0, 2)
    if num_sen_par != size(dp, 2)
        error("Same number of perturbations of initial conditions and parameters required")
    end
    if size(du0, 1) != length(u0)
        error("Perturbations for all initial conditions required")
    end
    if size(dp, 1) != length(p)
        error("Perturbations for all parameters required")
    end

    pdual = ForwardDiff.Dual{
        typeof(ForwardDiff.Tag(f, eltype(vec(p)))),
    }.(p,
        [ntuple(j -> dp[i, j], num_sen_par) for i in eachindex(p)])
    u0dual = ForwardDiff.Dual{
        typeof(ForwardDiff.Tag(f, eltype(vec(u0)))),
    }.(u0,
        [ntuple(j -> du0[i, j], num_sen_par)
         for i in eachindex(u0)])

    if (convert_tspan(alg) === nothing &&
        haskey(kwargs, :callback) && has_continuous_callback(kwargs.callback)) ||
       (convert_tspan(alg) !== nothing && convert_tspan(alg))
        tspandual = convert.(eltype(pdual), tspan)
    else
        tspandual = tspan
    end

    prob_dual = ODEProblem(f, u0dual, tspan, pdual,
        ODEForwardSensitivityProblem{DiffEqBase.isinplace(f),
            typeof(alg)}(alg);
        kwargs...)
end

"""
extract_local_sensitivities

Extracts the time series for the local sensitivities from the ODE solution. This requires
that the ODE was defined via `ODEForwardSensitivityProblem`.

```julia
extract_local_sensitivities(sol, asmatrix::Val = Val(false)) # Decompose the entire time series
extract_local_sensitivities(sol, i::Integer, asmatrix::Val = Val(false)) # Decompose sol[i]
extract_local_sensitivities(sol, t::Union{Number, AbstractVector},
                            asmatrix::Val = Val(false)) # Decompose sol(t)
```
"""
function extract_local_sensitivities(sol, asmatrix::Val = Val(false))
    extract_local_sensitivities(sol, sol.prob.problem_type.sensealg, asmatrix)
end
function extract_local_sensitivities(sol, asmatrix::Bool)
    extract_local_sensitivities(sol, Val{asmatrix}())
end
function extract_local_sensitivities(sol, i::Integer, asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol[i], asmatrix)
end
function extract_local_sensitivities(sol, i::Integer, asmatrix::Bool)
    extract_local_sensitivities(sol, i, Val{asmatrix}())
end
function extract_local_sensitivities(sol, t::Union{Number, AbstractVector},
    asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol(t), asmatrix)
end
function extract_local_sensitivities(sol, t, asmatrix::Bool)
    extract_local_sensitivities(sol, t, Val{asmatrix}())
end
function extract_local_sensitivities(tmp, sol, t::Union{Number, AbstractVector},
    asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol(tmp, t), asmatrix)
end
function extract_local_sensitivities(tmp, sol, t, asmatrix::Bool)
    extract_local_sensitivities(tmp, sol, t, Val{asmatrix}())
end

function extract_local_sensitivities(sol, ::ForwardDiffSensitivity, ::Val{false})
    u = ForwardDiff.value.(sol)
    du_full = ForwardDiff.partials.(sol)
    firststate = first(du_full)
    firstparam = first(firststate)
    Js = map(1:length(firstparam)) do j
        map(CartesianIndices(du_full)) do II
            du_full[II][j]
        end
    end
    return u, Js
end

function extract_local_sensitivities(sol, ::ForwardDiffSensitivity, ::Val{true})
    retu = ForwardDiff.value.(sol)
    jsize = length(sol.u[1]), ForwardDiff.npartials(sol.u[1][1])
    du = map(sol.u) do u
        du_i = similar(retu, jsize)
        for i in eachindex(u)
            du_i[i, :] = ForwardDiff.partials(u[i])
        end
        du_i
    end
    retu, du
end

# Get ODE u vector and sensitivity values from sensitivity problem u vector
function _extract(sol, sensealg::ForwardDiffSensitivity, su::AbstractVector,
    asmatrix::Val = Val(false))
    u = ForwardDiff.value.(su)
    du = _extract_du(sol, sensealg, su, asmatrix)
    return u, du
end

# Get sensitivity values from sensitivity problem u vector (nested form)

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{false})
    du_full = ForwardDiff.partials.(su)
    return [[du_full[i][j] for i in 1:size(du_full, 1)] for j in 1:length(du_full[1])]
end

# Get sensitivity values from sensitivity problem u vector (matrix form)

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{true})
    du_full = ForwardDiff.partials.(su)
    return [du_full[i][j] for i in 1:size(du_full, 1), j in 1:length(du_full[1])]
end

### Bonus Pieces

function SciMLBase.remake(prob::ODEProblem{uType, tType, isinplace, P, F, K,
        <:ODEForwardSensitivityProblem};
    f = nothing, tspan = nothing, u0 = nothing, p = nothing,
    kwargs...) where
    {uType, tType, isinplace, P, F, K}
    _p = p === nothing ? prob.p : p
    _f = f === nothing ? prob.f.f : f
    _u0 = u0 === nothing ? prob.u0[1:(prob.f.numindvar)] : u0[1:(prob.f.numindvar)]
    _tspan = tspan === nothing ? prob.tspan : tspan
    ODEForwardSensitivityProblem(_f, _u0,
        _tspan, _p, prob.problem_type.sensealg;
        prob.kwargs..., kwargs...)
end
