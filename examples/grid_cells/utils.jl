using ADSP, DifferentialEquations

## Set general parameters ##

num_paths=50                                                # generate 50 paths to show in the spagghetti plot
trange = (0.0, 0.3)                                         # time duration of each simulation run
path_params = (v̄ = 0.25, γ = 10.0, aₐ= 0.5, aᵥ= 0.1)        # parameters determining the paths dynamics 
grid_params = (r = 13/45*0.1, xscale=0.1, yscale=0.1*85/90)     # parameters determining the aspect and distance of the hex-grid
domain = ((0,0),(grid_params.xscale,grid_params.yscale))    # set the box domain containing the hex grid

# fix position, radii and associated color of each gridcell population
gridcell_colors = ["#fcaf3e", "#8ae234", "#ad7fa8", "#729f7e", "#fce94f", "#729fcf", "#ef2929", "#cccccc"]

## Construct receptive field populations ##
λ = 20.0 #[Hz] rate at which population spikes are emitted

# grid cell centers are located on a hexagonal grid (here 6 centers on a circle around the 1st)
gridcell_centers = [[[0.5*grid_params.xscale,0.5*grid_params.yscale]];[[cos(2π*α)*grid_params.r+0.5*grid_params.xscale,sin(2π*α)*grid_params.r+0.5*grid_params.yscale] for α ∈ 0:1//6:5//6]]
# the grid cells have a "radius" corresponding to the standard deviation of the Gaussian function
gridcell_radii = fill(grid_params.r/3, length(gridcell_centers))
# the receptive field functions are Gaussians centered at the gridcell_centers with radii determined by gridcell_radii
rfs = [v->exp(-sum((v[1:2].-μ).^2)/(2σ^2))  for (μ,σ) ∈ zip(gridcell_centers, gridcell_radii)]

"""
    generatePath(trange, params, u₀; dt=1e-3)

Generate one paths in the time interval `trange` with parameters specified by `param` and an initial condition `u₀` according to an sODE. 
`params` must contain:
    `v̄`    the average velocity of the path
    `γ`    the convergence rate with which the momentary velocity relaxes back to the `v̄`
    `aₐ`   the scale of acceleration in the angle component
    `aᵥ`   the scale of acceleration in the velocity component

Return a JuliaDiffEq `solution` object.
"""
function generatePath(trange, params, u₀; dt=1e-3)
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
    sol = solve(prob, SRIW1(), dt=dt)

    return sol
end

# draw an initial condition from the bounded region
generate_u₀(params,domain) = [rand()*(domain[2][1]-domain[1][1])+domain[1][1], rand()*(domain[2][2]-domain[1][2])+domain[1][2], rand(), randn()*params.aᵥ/√(2*params.γ)+params.v̄]


function run_path(cfg, log; extra_events=[], group_size=10, background_rate=0.0)
    # instantiate the network and a corresponding logger
    net = open(cfg) do f
        load_yaml(Network{Float64}, f)
    end

    # generate a path, convert it into spike trains and convert the spike trains into Events
    p = generatePath(trange, path_params, generate_u₀(path_params, domain))
    s = generateRFSpikes(trange, p, rfs, λ; group_size=fill(group_size,7), background_rate=background_rate)
    events = sort!([generateEventsFromSpikes(s, [:b,:e,:c,:g,:f,:a,:d]; spike_duration=0.005);extra_events])

    # simulate the neuron's response to the input
    log=simulate!(net, events;log=log, show_progress=false)

    return p,s,log
end

function estimate_response_probabilities(N;n_pop=20,n_ens=10,θ=5,p_trans=0.5,p_pop=0.5, λ=1.0)
    # Back of the envelope calculation:
    # Probability of a neuron's segment turning on (threshold ≧θ spikes) after one population spike (20 neurons, 50% corr., 50% transm.)
    P1 = 1.0-cdf(Binomial(n_pop,p_trans*p_pop),θ-1)
    
    # Probability of the segment turning on after multiple spikes, drawn from a Poisson distribution with rate λ
    S = rand(Poisson(λ),100000)
    P2 = mean(1.0.-(1.0-P1).^S)
    
    # Probability of a neuron (i.e. all 3 detector segments) turning on after a sequence of population spikes for each population
    P3 = P2^3

    # Probability that at least N out of 10 neurons would turn
    P4 = 1.0.-cdf.(Binomial(n_ens, P3),N.-1)
    
    return P4
end