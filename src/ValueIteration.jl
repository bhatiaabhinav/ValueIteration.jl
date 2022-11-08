module ValueIteration

using MDPs

export bellman_backup_synchronous, policy_evaluation, value_iteration, PolicyEvaluationHook

function bellman_backup_synchronous(mdp::AbstractMDP{Int, Int}, q::Matrix{Float64}, v::Vector{Float64})
    γ = discount_factor(mdp)
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

function policy_evaluation(mdp::AbstractMDP{Int, Int}, π::AbstractPolicy{Int, Int}; ϵ=0.01, infinite_horizon=false)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}
    nstates = length(state_space(mdp))
    nactions = length(action_space(mdp))
    𝕊 = state_space(mdp)
    𝔸 = action_space(mdp)

    q = zeros(nactions, nstates)
    v = zeros(nstates)

    for i in 1:(infinite_horizon ? typemax(Int) : horizon(mdp))
        δ = bellman_backup_synchronous(mdp, q, v)
        v .= map(s -> sum(a -> π(s, a) * q[a, s], 𝔸), 𝕊)
        if δ < ϵ
            # println("iter $i: breaking because δ=$δ < ϵ=$ϵ")
            break
        end
    end

    J = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


function value_iteration(mdp::AbstractMDP{Int, Int}; ϵ=0.01, infinite_horizon=false)::Tuple{Float64, Vector{Float64}, Matrix{Float64}}
    nstates = length(state_space(mdp))
    nactions = length(action_space(mdp))
    𝕊 = state_space(mdp)

    q = zeros(nactions, nstates)
    v = zeros(nstates)

    for i in 1:(infinite_horizon ? typemax(Int) : horizon(mdp))
        δ = bellman_backup_synchronous(mdp, q, v)
        v .= transpose(maximum(q, dims=1))
        # println(δ)
        if δ < ϵ
            # println("iter $i: breaking because δ=$δ < ϵ=$ϵ")
            break
        end
    end

    J = sum(start_state_distribution(mdp, 𝕊) .* v)

    return J, v, q
end


struct PolicyEvaluationHook <: AbstractHook
    n::Int
    returns::Vector{Float64}
    PolicyEvaluationHook(n) = new(n, Float64[])
end

function MDPs.preexperiment(peh::PolicyEvaluationHook; env, policy, kwargs...)
    push!(peh.returns, policy_evaluation(env, policy)[1])
end

function MDPs.postepisode(peh::PolicyEvaluationHook; env, policy, returns, kwargs...)
    if length(returns) % peh.n == 0
        push!(peh.returns, policy_evaluation(env, policy)[1])
    end
end


end # module ValueIteration
