module ValueIteration

using MDPs

export policy_evaluation, value_iteration, policy_iteration, PolicyEvaluationHook

function bellman_backup_synchronous(mdp::AbstractMDP{S, A}, q::Matrix{Float64}, v::Vector{Float64}, γ::Real) where {S, A}
    @inline R(s,a,s′) = reward(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
    @inline T(s,a,s′) = transition_probability(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
    𝕊 = state_space(mdp)
    𝔸 = action_space(mdp)
    δ = 0
    for s in eachindex(𝕊)
        for a in eachindex(𝔸)
            qᵢ₊₁ = sum(s′ -> T(s, a, s′)*(R(s, a, s′) + γ * v[s′]), transition_support(mdp, 𝕊[s], 𝔸[a]))
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

- `J`: The optimal value over the start state distribution.
- `v`: The optimal value function.
- `q`: The optimal Q function.
"""
function policy_evaluation(mdp::AbstractMDP{S, A}, π::AbstractPolicy{_S, _A}, γ::Real, horizon::Real; ϵ=0.01)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A, _S, _A}
    nstates = length(state_space(mdp))
    nactions = length(action_space(mdp))
    𝕊 = state_space(mdp)
    𝔸 = action_space(mdp)

    q = zeros(nactions, nstates)
    v = zeros(nstates)

    _π(s, a) = π(_S == Int ? s : 𝕊[s], _A == Int ? a : 𝔸[a])

    i = 0
    while i < horizon
        δ = bellman_backup_synchronous(mdp, q, v, γ)
        v .= map(s -> sum(a -> _π(s, a) * q[a, s], eachindex(𝔸)), eachindex(𝕊))
        i += 1
        if δ < ϵ
            break
        end
    end

    J = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


"""
    value_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A}

Perform value iteration on the given MDP. The state and action spaces must be `IntegerSpace` or `EnumerableTensorSpace`.

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
function value_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A}
    nstates::Int = length(state_space(mdp))
    nactions::Int = length(action_space(mdp))
    𝕊::Union{IntegerSpace, EnumerableTensorSpace} = state_space(mdp)
    𝔸::Union{IntegerSpace, EnumerableTensorSpace} = action_space(mdp)

    q::Matrix{Float64} = isnothing(prealloc_q) ? zeros(nactions, nstates) : prealloc_q
    @assert size(q) == (nactions, nstates)
    v::Vector{Float64} = isnothing(prealloc_v) ? zeros(nstates) : prealloc_v
    @assert length(v) == nstates
    T::Array{Float64, 3} = isnothing(prealloc_T) ? zeros(nstates, nactions, nstates) : prealloc_T
    @assert size(T) == (nstates, nactions, nstates)
    R::Array{Float64, 3} = isnothing(prealloc_R) ? zeros(nstates, nactions, nstates) : prealloc_R
    @assert size(R) == (nstates, nactions, nstates)
    for s::Int in eachindex(𝕊)
        for a::Int in eachindex(𝔸)
            for s′::Int in eachindex(𝕊)
                @inbounds T[s′, a, s] = transition_probability(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
                @inbounds R[s′, a, s] = reward(mdp, 𝕊[s], 𝔸[a], 𝕊[s′])
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


"""
    policy_iteration(mdp::AbstractMDP{S, A}, γ::Real, horizon::Real; ϵ=0.01, prealloc_q::Union{Matrix{Float64}, Nothing}=nothing, prealloc_v::Union{Vector{Float64}, Nothing}=nothing, prealloc_T::Union{Array{Float64, 3}, Nothing}=nothing, prealloc_R::Union{Array{Float64, 3}, Nothing}=nothing)::Tuple{Float64, Vector{Float64}, Matrix{Float64}} where {S, A}

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
