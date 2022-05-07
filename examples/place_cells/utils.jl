using StochasticDiffEq

## Set general parameters ##

## Define the parameters for the experiments:

# Path parameters:
path_params = ( 
    v̄ = 0.5,   # the average velocity of the path
    γ = 10.0,   # the convergence rate with which the momentary velocity relaxes back to the `v̄`
    aₐ= 0.25,   # the scale of acceleration in the angle component
    aᵥ= 0.1     # the scale of acceleration in the velocity component
)

# Grid parameters:
grid_params = (
    r = 13/45*0.1,      # aspect ratio of the grid
    xscale=0.1,
    yscale=0.1*85/90
)

# Plotting domain:
domain = ((0,0),(grid_params.xscale,grid_params.yscale))    # set the box domain containing the hex grid

# grid cell centers are located on a hexagonal grid (here 6 centers on a circle around the 1st)
gridcell_centers = [
    [cos(2π*4/6)*grid_params.r+0.5*grid_params.xscale,sin(2π*4/6)*grid_params.r+0.5*grid_params.yscale],
    [                          0.5*grid_params.xscale,                          0.5*grid_params.yscale], 
    [cos(2π*1/6)*grid_params.r+0.5*grid_params.xscale,sin(2π*1/6)*grid_params.r+0.5*grid_params.yscale], 
]

λ = 250.0                                #[Hz] rate at which population spikes are emitted
λ_background = 10.0                     #[Hz] rate at which population spikes are emitted
t_jitter = 5.0e-3
path_trange = (0.0, 0.1)              # duration of generated path
r = 1.5*grid_params.r                   # distance of path start-point from center
v_opt = 2r/(path_trange[2]-path_trange[1])  # velocity of the generated path (set such that path is symmetric around middle)
trials = 500                            # number of trials over which to average the response probability

## Define helper functions 

"""
generate_stochastic_path(trange, params, u₀; dt=1e-3)

Generate one paths in the time interval `trange` with parameters specified by `param` and an initial condition `u₀` according to an sODE. 
`params` must contain:
`v̄`    the average velocity of the path
`γ`    the convergence rate with which the momentary velocity relaxes back to the `v̄`
`aₐ`   the scale of acceleration in the angle component
`aᵥ`   the scale of acceleration in the velocity component

Return a JuliaDiffEq `solution` object.
"""
function generate_stochastic_path(trange, params, u₀, args...; kwargs...)
    """
    Drift term of the movement.
    """
    function f(du, u, p, t)
        du[1] = cos(2π*u[3])*u[4]
        du[2] = sin(2π*u[3])*u[4]
        du[3] = 0.0    # the angle doesn't change deterministically
        du[4] = p.γ*(p.v̄-u[4])# model smooth changes in velocity
    end

    """
    Diffusion term of the movement.
    """
    function g(du, u,p,t)
        du[1] = 0.0     # x positions don't change randomly
        du[2] = 0.0     # y positions don't change randomly
        du[3] = p.aₐ      # model fluctuations in angle
        du[4] = p.aᵥ      # model fluctuations in velocity
    end

    prob = SDEProblem(f, g, u₀, trange, params)
    sol = solve(prob, SRIW1(), args...; kwargs...)

    return (t, args...; idxs=1:2, kwargs...) -> trange[1] ≤ t ≤ trange[2] ? sol(t, args...; idxs, kwargs...) : missing
end

"""
generate_straight_path(trange, α, v, x₀; kwargs...)

Generate one straight-line path in the time interval `trange` with angle `α` (c.c.w from positive x-axis, in radians), velocity `v` and starting position `x₀`.
Returns a function `t -> [x(t),y(t)]`.
"""
generate_straight_path(trange, α, v, x₀) = ((t)-> trange[1] ≤ t ≤ trange[2] ? x₀ .+ [cos(α),sin(α)] .* (v*(t-trange[1])) : missing)

"""
generate_u₀(params,domain)

draw an initial condition from the bounded `domain` according to the specified `params`.
"""
generate_u₀(params,domain) = [rand()*(domain[2][1]-domain[1][1])+domain[1][1], rand()*(domain[2][2]-domain[1][2])+domain[1][2], rand(), randn()*params.aᵥ/√(2*params.γ)+params.v̄]

"""generate_path_spikes(path, populations, λ; group_size=ones(Int,length(rfs)), dt=0.0, background_rate=0.0, t_jitter=0.0)

Generates outputs resembling the activity of multiple homogeneous neuron `populations`.
`populations` must be an iterable of objects with fields:
- `rf` for the population's receptive field
- `neurons` for the population's neurons.

Emits volleys of spikes at a constant rate `λ` and encodes the receptive field responses
`populations[i].rf(input(t))` to the input at those times in the magnitudes of the volleys.
The spikes are jittered by a random time-lag uniformly drawn from `[0,t_jitter]`.
"""
function generate_path_spikes(trange, path, populations, λ; dt=0.0, background_rate=0.0, t_jitter=0.0)
    T = (trange[end]-trange[1])

    all_spikes = Event{Float64}[]
    for population ∈ populations
        N = rand(Poisson(λ*T))
        master_spikes = rand(N) .* T .+ trange[1]
        
        p_accept = population.rf.(path.(master_spikes))
        
        for neuron in population.neurons
            background_spikes = rand(rand(Poisson(background_rate*T))) .* T .+ trange[1]
            t_accepted = master_spikes[rand(length(master_spikes)) .≤ p_accept]
            spikes = sort!([t_accepted .+ t_jitter .* rand(length(t_accepted)); background_spikes])
            dels = Int[]
            for (k,t) ∈ enumerate(spikes[2:end])
                if t ≤ spikes[k] + dt
                    push!(dels, k+1)
                end
            end
            deleteat!(spikes, dels)
            append!(all_spikes, Event.(:input_spikes, 0.0, spikes, Ref(neuron)))
        end
    end

    return all_spikes
end

"""run_path!(net, trange, path, populations, λ; extra_events=[], kwargs...)

Simulates one traversal of the `path` by the given `net` over a time-interval `trange`.
Spikes are generated for each population in `populations` in volleys at fixed rate `λ`.
"""
function run_path!(net, trange, path, populations, λ; extra_events=Event{Float64}[], kwargs...)
    DPC.reset!(net)
    
    # convert path into spike trains and convert the spike trains into Events
    s = generate_path_spikes(trange, path, populations, λ; kwargs...)
    events = sort!([s; extra_events])

    # simulate the neuron's response to the input
    logger=simulate!(net, events)

    return s,logger
end