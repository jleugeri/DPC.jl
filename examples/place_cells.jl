using ADSP, CairoMakie, ProgressMeter, Distributions, AbstractPlotting.MakieLayout

#include(joinpath(@__DIR__, "utils.jl"))
include("utils.jl")
include(joinpath(@__DIR__, "place_cells", "utils.jl"))

## Initialize the network
(net,objects) = load_network(joinpath(@__DIR__, "place_cells", "network.yaml"), weight_type=BernoulliSynapseWeight{Float64})

# Set up the place-cell populations:
populations = [
    (
        # the receptive field functions are Gaussians centered at the gridcell_centers with radii determined by gridcell_radii
        rf = v-> ismissing(v) ? 0.0 : exp(-sum((v.-μ).^2)/(2*(grid_params.r/3)^2)),
        neurons = n
    ) for (μ,n) in zip(gridcell_centers,[[objects[Symbol("i$(i)$(lpad(j,2,"0"))")] for j in 1:20] for i in 1:3])
]


## Randomly sample stochastic paths 
num_paths = 50
prog = Progress(num_paths+1, 1, "Sampling paths...")
i=0
tt = LinRange(path_trange..., 50)
ProgressMeter.update!(prog, i)
paths = Vector{NamedTuple{(:x,:y),Tuple{Vector{Float64},Vector{Float64}}}}(undef, num_paths)
special_path = nothing
while i < num_paths
    # generate a single path and the response
    path = generate_stochastic_path(path_trange, path_params, generate_u₀(path_params, domain))
    spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter)

    # if this path resulted in a spike, save it
    if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
        global i+=1
        paths[i] = (x=path.(tt,idxs=1),y=path.(tt,idxs=2))

        if i == 1
            global special_path = (;path=paths[i], spikes, logger)
        end
    end
    ProgressMeter.update!(prog, i)
end


plateau_starts_1 = filter(x->(x.object==:seg1 && x.event==:plateau_starts), special_path.logger.data)
plateau_starts_2 = filter(x->(x.object==:seg2 && x.event==:plateau_starts), special_path.logger.data)
plateau_extended_1 = filter(x->(x.object==:seg1 && x.event==:plateau_extended), special_path.logger.data)
plateau_extended_2 = filter(x->(x.object==:seg2 && x.event==:plateau_extended), special_path.logger.data)
spike_times = filter(x->(x.object==:n && x.event==:spikes), special_path.logger.data)

## Vary the speed
α_opt = 2π/6
x₀_opt = [cos(α_opt+π)*r+0.5*grid_params.xscale,sin(α_opt+π)*r+0.5*grid_params.yscale]
vs = LinRange(v_opt/100,2*v_opt,100)

counts = zeros(Int, length(vs))
@showprogress 1.0 "Sweeping run-speeds ..." for (i,v) ∈ enumerate(vs)
    path = generate_straight_path((path_trange[1],path_trange[1]+2r/v), α_opt, v, x₀_opt)
    for t ∈ 1:trials
        spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter)

        if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
            counts[i] += 1
        end
    end
end
prob_speed = counts ./ trials


## Vary the orientation
αs = LinRange(0,2π,101)[1:end-1]        # movement directions over which to sweep
x₀s_rotated = [[cos(α+π)*r+0.5*grid_params.xscale,sin(α+π)*r+0.5*grid_params.yscale] for α ∈ αs] # path starting points
counts = zeros(Int, length(αs))
@showprogress 1.0 "Sweeping directions ..." for (i,(α,x₀)) ∈ enumerate(zip(αs, x₀s_rotated))
    path = generate_straight_path(path_trange, α, v_opt, x₀)
    for t ∈ 1:trials
        spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter)

        if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
            counts[i] += 1
        end
    end
end
prob_rotated = counts ./ trials
hist_rotated = (hcat(cos.(αs),sin.(αs)) .*r .* prob_rotated) .+ 0.5 .* [grid_params.xscale grid_params.yscale]


## Vary the shift
offsets = LinRange(-2r,2r,51)
x₀s_offset = eachrow(x₀_opt' .+ [cos(α_opt+π/2) sin(α_opt+π/2)] .* offsets)
counts = zeros(Int, length(offsets))
@showprogress 1.0 "Sweeping offsets ..." for (i,x₀) ∈ enumerate(x₀s_offset)
    path = generate_straight_path(path_trange, α_opt, v_opt, x₀)
    for t ∈ 1:trials
        spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter)

        if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
            counts[i] += 1
        end
    end
end
prob_offset = counts ./ trials
hist_offset = [cos(α_opt+π/2) sin(α_opt+π/2)].*offsets .+ [cos(α_opt) sin(α_opt)] .* prob_offset .*r .+ 0.5 .* [grid_params.xscale grid_params.yscale]


## Plotting
# Set up plots
show_spines = (
    :leftspinevisible => true,
    :rightspinevisible => true,
    :bottomspinevisible => true,
    :topspinevisible => true
)

fig = Figure(resolution = (800, 550))
ax11 = fig[1,1] = Axis(fig; title="Effective paths", backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax11)
gl = fig[1,2:3] = GridLayout()
ax121 = gl[1,1] = Axis(fig, title="Expected volley sizes for the highlighted path")
ax122 = gl[2,1] = Axis(fig, title="Activity for the highlighted path", xlabel="time [ms]")
hidexdecorations!(ax121, grid=false)
hideydecorations!(ax122, label=false)
ax14 = fig[1,4] = Axis(fig; aspect=DataAspect(), backgroundcolor=:transparent)
hidedecorations!(ax14)
ax21 = fig[2,1] = Axis(fig; title="Optimal path", backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax21)
ax22 = fig[2,2] = Axis(fig, title="Speed", xlabel="run speed [m/s]", ylabel="spike probability")
ax23 = fig[2,3] = Axis(fig; title="Orientation", backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax23)
ax24 = fig[2,4] = Axis(fig; title="Translation", backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax24)

rowsize!(fig.layout, 1, Aspect(1, grid_params.yscale/grid_params.xscale))
rowsize!(fig.layout, 2, Aspect(1, grid_params.yscale/grid_params.xscale))

linkaxes!(ax11, ax21)
linkaxes!(ax11, ax23)
linkaxes!(ax11, ax24)

linkxaxes!(ax121, ax122)


# Plot the stochastically sampled paths
pop!(paths)
for path in paths
    lines!(ax11, path.x, path.y, linewidth=2, color=RGBAf0(0.2,0.2,0.2,0.5))
end
lines!(ax11, p1.x, p1.y, linewidth=4, color=color_1)

# Plot volley sizes
for (pop,color) in zip(populations,gridcell_colors)
    volley_size=pop.rf.(zip(special_path.path.x, special_path.path.y)).*length(pop.neurons)
    band!(ax121, tt, 0, volley_size; color=RGBAf0(RGB(to_color(color)),0.5))
end

# Plot the spikes, plateaus and rectangles to highlight plateau triggers
seg1 = get_trace(:seg1, special_path.logger.data)
steps!(ax122, [0;seg1.t;300e-3], 19.9*[0;Int.(seg1.state).>1;0] .- 20.05, color=:transparent, fill=RGBAf0(RGB(to_color(gridcell_colors[1])),0.5))
seg2 = get_trace(:seg2, special_path.logger.data)
steps!(ax122, [0;seg2.t;300e-3], 19.9*[0;Int.(seg2.state).>1;0] .- 40.05, color=:transparent, fill=RGBAf0(RGB(to_color(gridcell_colors[2])),0.5))
for (i,color) in enumerate(gridcell_colors)
    for j in 1:20
        epsp = get_trace(Symbol("syn$(i)$(lpad(j,2,"0"))"), special_path.logger.data)
        steps!(ax122, [0;epsp.t;300e-3],-(i-1)*20 .+ -j .+ 0.9 .* [0;Int.(epsp.state);0], color=:transparent, fill=color)
    end
end
lines!(ax122, Rect(plateau_starts_1.t[]-5.5e-3, -20.25, 11e-3, 20.5), color=:gray10, linewidth=2)
for t in plateau_extended_1.t
    lines!(ax122, [t,t], [-20.25, -0.25], color=:gray10, linewidth=2)
end
lines!(ax122, Rect(plateau_starts_2.t[]-5.5e-3, -40.25, 11e-3, 20.5), color=:gray10, linewidth=2)
for t in plateau_extended_2.t
    lines!(ax122, [t,t], [-40.25, -20.25], color=:gray10, linewidth=2)
end
lines!(ax122, Rect(spike_times.t[1]-5.5e-3, -60.25, 11e-3, 20.5), color=:gray10, linewidth=2)

# Plot the neuron
plot!(ax14, objects[:n],
    branch_width=1, 
    branch_length=8.0, 
    color=Dict(:n=>color_3, :seg1=>color_2, :seg2=>color_1)
)

# Plot the optimal path
path_opt = generate_straight_path(path_trange, α_opt, v_opt, x₀_opt)
(path_start,path_end) = path_opt.(path_trange)
arrows!(ax21, Point2f0[path_start], Point2f0[path_end .- path_start], linewidth=2, color=:black, arrowsize=10)

# Plot the speed-dependent spike probability
lines!(ax22, vs, prob_speed, color=color_1, linewidth=2)
vlines!(ax22, [v_opt], linestyle=:dash, color=:gray, linewidth=2)

# Plot the orientation-dependent spike probability
lines!(ax23, decompose(Point2f0,Circle(Point2f0(grid_params.xscale/2,grid_params.yscale/2), 0.5*r)), color=:gray)
lines!(ax23, decompose(Point2f0,Circle(Point2f0(grid_params.xscale/2,grid_params.yscale/2), r)), color=:gray)

for (p,α,x₀) in zip(prob_rotated, αs, x₀s_rotated)
    path = generate_straight_path(path_trange, α, v_opt, x₀)
    (path_start,path_end) = path.(path_trange)
    arrows!(ax23, Point2f0[path_start], Point2f0[path_end .- path_start], linewidth=2, linecolor=RGBAf0(0.2,0.2,0.2,p), arrowcolor=RGBAf0(0.2,0.2,0.2,p), arrowsize=5)
end
poly!(ax23, Point2f0.(eachrow(hist_rotated)), color=color_1_50)
lines!(ax23, Point2f0.(eachrow(hist_rotated)), linewidth=2, color=color_1)


# Plot the offset-dependent spike probability
lines!(ax24, 
    0.5 .* Point2f0[(grid_params.xscale, grid_params.yscale)].+ Point2f0[(cos(α_opt+π/2), sin(α_opt+π/2)),(-cos(α_opt+π/2), -sin(α_opt+π/2))].*2r
    , color=:gray
)
lines!(ax24, 
    0.5 .* Point2f0[(grid_params.xscale, grid_params.yscale)].+ Point2f0[(cos(α_opt), sin(α_opt))].*0.5r .+ Point2f0[(cos(α_opt+π/2), sin(α_opt+π/2)),(-cos(α_opt+π/2), -sin(α_opt+π/2))].*2r
    , color=:gray
)
lines!(ax24, 
    0.5 .* Point2f0[(grid_params.xscale, grid_params.yscale)].+ Point2f0[(cos(α_opt), sin(α_opt))].*r .+ Point2f0[(cos(α_opt+π/2), sin(α_opt+π/2)),(-cos(α_opt+π/2), -sin(α_opt+π/2))].*2r
    , color=:gray
)

lines!(ax24, hist_offset)
for (p,x₀) in zip(prob_offset, x₀s_offset)
    path = generate_straight_path(path_trange, α_opt, v_opt, x₀)
    (path_start,path_end) = path.(path_trange)
    arrows!(ax24, Point2f0[path_start], Point2f0[path_end .- path_start], linewidth=2, linecolor=RGBAf0(0.2,0.2,0.2,p), arrowcolor=RGBAf0(0.2,0.2,0.2,p), arrowsize=5)
end
poly!(ax24, Point2f0.(eachrow(hist_offset)), color=color_1_50)
lines!(ax24, Point2f0.(eachrow(hist_offset)), linewidth=2, color=color_1)

xlims!(ax11, (domain[1][1],domain[2][1]))
ylims!(ax11, domain[1][2],domain[2][2])

save(joinpath("figures","place_cells.pdf"), fig)
save(joinpath("figures","place_cells.svg"), fig)
save(joinpath("figures","place_cells.png"), fig)
fig