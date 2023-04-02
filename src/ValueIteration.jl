module ValueIteration

using MDPs

export policy_evaluation, value_iteration, PolicyEvaluationHook

function bellman_backup_synchronous(mdp::AbstractMDP{Int, Int}, q::Matrix{Float64}, v::Vector{Float64}, γ::Real)
    @inline R(s,a,s′) = reward(mdp, s, a, s′)
    @inline T(s,a,s′) = transition_probability(mdp, s, a, s′)
    nactions, nstates = size(q)
    δ = 0
    for s in 1:nstates
        for a in 1:nactions
            qᵢ₊₁ = sum(s′ -> T(s, a, s′)*(R(s, a, s′) + γ * v[s′]), transition_support(mdp, s, a))
            δ = max(δ, abs(qᵢ₊₁ - q[a, s]))
            q[a, s] = qᵢ₊₁
        end
    end
    return δ
end

function bellman_backup_synchronous!(q::Matrix{Float64}, v::Vector{Float64}, R::Array{Float64, 3}, T::Array{Float64, 3}, γ::Float64)::Float64
    nactions::Int, nstates::Int = size(q)
    δ::Float64 = 0
    for s::Int in 1:nstates
        @inbounds for a::Int in 1:nactions
            qᵢ₊₁::Float64 = 0
            @simd for s′ in 1:nstates
                qᵢ₊₁ += T[s′, a, s]*(R[s′, a, s] + γ * v[s′])
            end
            δₐₛ::Float64 = abs(qᵢ₊₁ - q[a, s])
            if δₐₛ > δ
                δ = δₐₛ
            end
            q[a, s] = qᵢ₊₁
        end
    end
    return δ
end

function policy_evaluation(mdp::AbstractMDP{Int, Int}, π::AbstractPolicy{Int, Int}, γ::Real, horizon::Real; ϵ=0.01)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}
    nstates = length(state_space(mdp))
    nactions = length(action_space(mdp))
    𝕊 = state_space(mdp)
    𝔸 = action_space(mdp)

    q = zeros(nactions, nstates)
    v = zeros(nstates)

    i = 0
    while i < horizon
        δ = bellman_backup_synchronous(mdp, q, v, γ)
        v .= map(s -> sum(a -> π(s, a) * q[a, s], 𝔸), 𝕊)
        i += 1
        if δ < ϵ
            # println("iter $i: breaking because δ=$δ < ϵ=$ϵ")
            break
        end
    end

    J = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


# function value_iteration(mdp::AbstractMDP{Int, Int}, γ::Real, horizon::Real; ϵ=0.01)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}
#     nstates = length(state_space(mdp))
#     nactions = length(action_space(mdp))
#     𝕊 = state_space(mdp)

#     q = zeros(nactions, nstates)
#     v = zeros(nstates)

#     # println("starting")
#     i = 0
#     while i < horizon
#         δ = bellman_backup_synchronous(mdp, q, v, γ)
#         v .= transpose(maximum(q, dims=1))
#         i += 1
#         if δ < ϵ
#             # println("iter $i: breaking because δ=$δ < ϵ=$ϵ")
#             break
#         end
#     end

#     J = sum(start_state_distribution(mdp, 𝕊) .* v)

#     return J, v, q
# end

"""
    value_iteration(mdp::AbstractMDP{Int, Int}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}

Perform value iteration on the given MDP.

# Arguments

- `mdp`: The MDP to perform value iteration on.
- `γ`: The discount factor.
- `horizon`: The maximum number of iterations to perform.
- `ϵ`: The convergence threshold.
- `prealloc_q`: A preallocated matrix to use for the Q function.
- `prealloc_v`: A preallocated vector to use for the V function.
- `prealloc_T`: A preallocated array to use for the transition probability function.
- `prealloc_R`: A preallocated array to use for the reward function.

# Returns

- `J`: The optimal value over the start state distribution.
- `v`: The optimal value function.
- `q`: The optimal Q function.
"""
function value_iteration(mdp::AbstractMDP{Int, Int}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))
    𝕊::IntegerSpace = state_space(mdp)

    q::Matrix{Float64} = isnothing(prealloc_q) ? zeros(nactions, nstates) : prealloc_q
    @assert size(q) == (nactions, nstates)
    v::Vector{Float64} = isnothing(prealloc_v) ? zeros(nstates) : prealloc_v
    @assert length(v) == nstates
    T::Array{Float64, 3} = isnothing(prealloc_T) ? zeros(nstates, nactions, nstates) : prealloc_T
    @assert size(T) == (nstates, nactions, nstates)
    R::Array{Float64, 3} = isnothing(prealloc_R) ? zeros(nstates, nactions, nstates) : prealloc_R
    @assert size(R) == (nstates, nactions, nstates)
    for s::Int in 1:nstates
        for a::Int in 1:nactions
            for s′::Int in 1:nstates
                @inbounds T[s′, a, s] = transition_probability(mdp, s, a, s′)
                @inbounds R[s′, a, s] = reward(mdp, s, a, s′)
            end
        end
    end
    γf64 = Float64(γ)

    i::Int = 0
    while i < horizon
        δ = bellman_backup_synchronous!(q, v, R, T, γf64)
        for s::Int in 1:nstates
            @inbounds v[s] = @views maximum(q[:, s])
        end
        i += 1
        δ < ϵ && break
    end

    J::Float64 = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


struct PolicyEvaluationHook <: AbstractHook
    π::AbstractPolicy
    γ::Real
    horizon::Real
    n::Int
    returns::Vector{Float64}
    PolicyEvaluationHook(π::AbstractPolicy, γ::Real, horizon::Real, n::Int) = new(π, γ, horizon, n, Float64[])
end

function MDPs.preexperiment(peh::PolicyEvaluationHook; env, kwargs...)
    push!(peh.returns, policy_evaluation(env, peh.π, peh.γ, peh.horizon)[1])
end

function MDPs.postepisode(peh::PolicyEvaluationHook; env, returns, kwargs...)
    if length(returns) % peh.n == 0
        push!(peh.returns, policy_evaluation(env, peh.π, peh.γ, peh.horizon)[1])
    end
end


end # module ValueIteration
