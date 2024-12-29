module ValueIteration

using MDPs

export policy_evaluation, value_iteration, policy_iteration, PolicyEvaluationHook


"""
    bellman_backup_action_values_synchronous!(q::AbstractMatrix{T}, v::AbstractVector{T}, R::AbstractArray{T, 3}, Tr::AbstractArray{T, 3}, γ::T)::T where T<:AbstractFloat

Perform a synchronous Bellman backup on the action values with formula:

qᵢ₊₁(a, s) <- Σₛ′ Tr(s′, a, s) * (R(s′, a, s) + γ * v(s′))

# Arguments

- `q`: The action values to update, with indices [a, s].
- `v`: The value function, with indices [s].
- `R`: The reward function, with indices [s', a, s].
- `Tr`: The transition probability function, with indices [s', a, s].
- `γ`: The discount factor.

# Returns

- `δ`: The maximum change in the action values.
"""
function bellman_backup_action_values_synchronous!(q::AbstractMatrix{T}, v::AbstractVector{T}, R::AbstractArray{T, 3}, Tr::AbstractArray{T, 3}, γ::T)::T where T<:AbstractFloat
    nactions::Int, nstates::Int = size(q)
    δ::T = 0
    for s::Int in 1:nstates
        @inbounds for a::Int in 1:nactions
            qᵢ₊₁::T = 0
            @simd for s′ in 1:nstates
                qᵢ₊₁ += Tr[s′, a, s]*(R[s′, a, s] + γ * v[s′])
            end
            δₐₛ::T = abs(qᵢ₊₁ - q[a, s])
            if δₐₛ > δ
                δ = δₐₛ
            end
            q[a, s] = qᵢ₊₁
        end
    end
    return δ
end

function bellman_backup_action_values_vectorized(q::AbstractMatrix{T}, v::AbstractVector{T}, R::AbstractArray{T, 3}, Tr::AbstractArray{T, 3}, γ::T)::Union{AbstractMatrix{T}, T} where T<:AbstractFloat
    tmp = @. Tr * (R + γ * v)
    q_new = @views sum(tmp, dims=1)[1, :, :]
    δ = maximum(abs.(q_new - q))
    return q_new, δ
end

"""
    policy_evaluation(mdp::AbstractMDP{S, A}, π::AbstractPolicy{_S, _A}, γ::Real, horizon::Real; ϵ=0.01)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A, _S, _A}

Perform policy evalution on the given MDP. The state and action spaces must be `IntegerSpace` or `EnumerableTensorSpace`.

# Arguments

- `mdp`: The MDP to perform value iteration on.
- `π`: The policy to evaluate.
- `γ`: The discount factor.
- `horizon`: The maximum number of iterations to perform.
- `ϵ`: The convergence threshold.

# Returns

- `J`: The value over the start state distribution, under the given policy.
- `v`: The value function, where `v[s]` is the value of state `s` under the given policy.
- `q`: The Q function, where `q[a, s]` is the value of taking action `a` in state `s` under the given policy.
"""
function policy_evaluation(mdp::AbstractMDP{S, A}, π::AbstractPolicy{_S, _A}, γ::Real, horizon::Real; ϵ=0.01)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A, _S, _A}
    nstates = length(state_space(mdp))
    nactions = length(action_space(mdp))
    𝕊 = state_space(mdp)
    𝔸 = action_space(mdp)
    _π(s, a) = π(_S == Int ? s : 𝕊[s], _A == Int ? a : 𝔸[a])

    q::Matrix{Float64} = zeros(nactions, nstates)
    v::Vector{Float64} = zeros(nstates)
    T::Array{Float64, 3} = zeros(nstates, nactions, nstates)
    R::Array{Float64, 3} = zeros(nstates, nactions, nstates)
    πas = zeros(nactions, nstates)
    Threads.@threads for s::Int in eachindex(𝕊)
        for a::Int in eachindex(𝔸)
            for s′::Int in eachindex(𝕊)
                @inbounds T[s′, a, s] = transition_probability(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
                @inbounds R[s′, a, s] = reward(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
            end
            πas = _π(s, a)
        end
    end
    γf64 = Float64(γ)

    i::int = 0
    while i < horizon
        δ = bellman_backup_action_values_synchronous!(q, v, R, T, γf64)
        @inbounds for s::Int in 1:nstates
            v[s] = 0
            for a::Int in 1:nactions  # why is @simd not working here? says "simd loop index must be a symbol"
                v[s] += πas[a, s] * q[a, s]
            end
        end
        i += 1
        δ < ϵ && break
    end

    J = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


"""
    value_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A}

Perform value iteration on the given MDP. The state and action spaces must be `IntegerSpace` or `EnumerableTensorSpace`. Preallocated arrays can be provided for the Q, V, T, and R functions. The indices for the Q function are [a, s], the indices for the T function are [s', a, s], and the indices for the R function are [s', a, s].

# Arguments

- `mdp`: The MDP to perform value iteration on.
- `γ`: The discount factor.
- `horizon`: The maximum number of iterations to perform.
- `ϵ`: The convergence threshold.
- `prealloc_v`: A preallocated vector to use for the v function.
- `prealloc_q`: A preallocated matrix to use for the q function, with indices [a, s].
- `prealloc_T`: A preallocated array to use for the transition probability function, with indices [s', a, s].
- `prealloc_R`: A preallocated array to use for the reward function, with indices [s', a, s].

# Returns

- `J`: The optimal value over the start state distribution.
- `v`: The optimal value function, where `v[s]` is the optimal value of state `s`.
- `q`: The optimal q function, where `q[a, s]` is the optimal value of taking action `a` in state `s`.
"""
function value_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A}
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))
    𝕊::Union{IntegerSpace, EnumerableTensorSpace} = state_space(mdp)
    𝔸::Union{IntegerSpace, EnumerableTensorSpace} = action_space(mdp)

    q::Matrix{Float64} = isnothing(prealloc_q) ? zeros(nactions, nstates) : prealloc_q
    @assert size(q) == (nactions, nstates)
    v::Vector{Float64} = isnothing(prealloc_v) ? zeros(nstates) : prealloc_v
    @assert length(v) == nstates
    Tr::Array{Float64, 3} = isnothing(prealloc_T) ? zeros(nstates, nactions, nstates) : prealloc_T
    @assert size(Tr) == (nstates, nactions, nstates)
    R::Array{Float64, 3} = isnothing(prealloc_R) ? zeros(nstates, nactions, nstates) : prealloc_R
    @assert size(R) == (nstates, nactions, nstates)
    Threads.@threads for s::Int in eachindex(𝕊)
        for a::Int in eachindex(𝔸)
            for s′::Int in eachindex(𝕊)
                @inbounds Tr[s′, a, s] = transition_probability(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
                @inbounds R[s′, a, s] = reward(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
            end
        end
    end

    v, q = value_iteration(Tr, R, γ, horizon; ϵ=ϵ, prealloc_v=v, prealloc_q=q)

    J::Float64 = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


"""
    value_iteration(Tr::Array{T, 3}, R::Array{T, 3}, γ::Real, horizon::Real; ϵ=0.01, prealloc_v::Union{Vector{T}, Nothing}=nothing, prealloc_q::Union{Matrix{T}, Nothing}=nothing)::Tuple{Vector{T}, Matrix{T}} where T<:AbstractFloat

Perform value iteration given the transition probability and reward functions. The transition function indices are [s', a, s] and the reward function indices are [s', a, s]. Notice that s' is the first index in the transition function and the reward function. A preallocated vector and matrix can be provided for the V and Q functions. The indices for the Q function are [a, s].

# Arguments

- `Tr`: The transition probability function, where `Tr[s', a, s]` is the probability of transitioning to state `s'` given that action `a` is taken in state `s`.
- `R`: The reward function, where `R[s', a, s]` is the reward of transitioning to state `s'` given that action `a` is taken in state `s`.
- `γ`: The discount factor.
- `horizon`: The maximum number of iterations to perform.
- `ϵ`: The convergence threshold.
- `prealloc_v`: A preallocated vector to use for the v function.
- `prealloc_q`: A preallocated matrix to use for the q function.

# Returns

- `v`: The optimal value function, where `v[s]` is the optimal value of state `s`.
- `q`: The optimal Q function, where `q[a, s]` is the optimal value of taking action `a` in state `s`.
"""
function value_iteration(Tr::Array{T, 3}, R::Array{T, 3}, γ::Real, horizon::Real; ϵ=0.01, prealloc_v::Union{Vector{T}, Nothing}=nothing, prealloc_q::Union{Matrix{T}, Nothing}=nothing)::Tuple{Vector{T}, Matrix{T}} where T<:AbstractFloat
    nstates::Int = size(Tr, 1)
    nactions::Int = size(Tr, 2)

    q::Matrix{Float64} = isnothing(prealloc_q) ? zeros(nactions, nstates) : prealloc_q
    @assert size(q) == (nactions, nstates)
    v::Vector{Float64} = isnothing(prealloc_v) ? zeros(nstates) : prealloc_v
    @assert length(v) == nstates
    @assert size(Tr) == (nstates, nactions, nstates)
    @assert size(R) == (nstates, nactions, nstates)
    γf64 = Float64(γ)

    i::Int = 0
    while i < horizon
        δ = bellman_backup_action_values_synchronous!(q, v, R, Tr, γf64)
        for s::Int in 1:nstates
            @inbounds v[s] = @views maximum(q[:, s])
        end
        i += 1
        δ < ϵ && break
    end

    return v, q
end


"""
    policy_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, policy_improvement_step = q -> GreedyPolicy(q), max_iterations=nothing)::Tuple{Vector{Float64}, Vector{Vector{Float64}}, Vector{Matrix{Float64}}, Vector{P}} where {S, A, P<:AbstractPolicy}

Perform policy iteration on the given MDP. The state and action spaces must be `IntegerSpace` or `EnumerableTensorSpace`.

# Arguments

- `mdp`: The MDP to perform value iteration on.
- `γ`: The discount factor.
- `horizon`: The maximum number of iterations to perform.
- `ϵ`: The convergence threshold.
- `policy_improvement_step`: The policy improvement step to use. Defaults to `q -> GreedyPolicy(q)`.
- `max_iterations`: The maximum number of iterations to perform. Defaults to `nothing`, which means that the algorithm will run until convergence.

# Returns

- `Js`: The optimal value over the start state distribution for each iteration.
- `vs`: The optimal value function for each iteration.
- `qs`: The optimal Q function for each iteration.
- `policies`: The policy for each iteration.
"""
function policy_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, policy_improvement_step = q -> GreedyPolicy(q), max_iterations=nothing) where {S, A}
    Js::Vector{Float64} = Float64[]
    vs::Vector{Vector{Float64}} = Vector{Float64}[]
    qs::Vector{Matrix{Float64}} = Matrix{Float64}[]
    policies = []

    policy_changed = true
    policy = GreedyPolicy(zeros(length(action_space(mdp)), length(state_space(mdp))))
    states = state_space(mdp) |> collect
    actions = policy.(states)
    iters = 0
    while policy_changed
        J, v, q = policy_evaluation(mdp, policy, γ, horizon; ϵ=ϵ)
        push!.((Js, vs, qs, policies), (J, v, q, policy))
        policy = policy_improvement_step(q)
        actions_after = policy.(states)
        # policy_changed = actions != actions_after
        actions = actions_after
        iters += 1
        if !isnothing(max_iterations) && iters >= max_iterations
            break
        end
    end
    
    return Js, vs, qs, policies
end

"""
    PolicyEvaluationHook(π::AbstractPolicy, γ::Real, horizon::Real, n::Int)

A hook for use with `MDPs.interact` that evaluates the given policy every `n` episodes and stores the result in `returns`.

# Arguments
- `π`: The policy to evaluate.
- `γ`: The discount factor.
- `horizon`: The horizon to use for policy evaluation.
- `n`: The number of episodes between evaluations.
"""
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
