using DPC, CairoMakie, ProgressMeter, Distributions, Makie.MakieLayout

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
    spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter, background_rate=λ_background)

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
vs = LinRange(v_opt/100,10*v_opt,100)

prob_speed = Vector{Float64}[]
prob_rotated = Vector{Float64}[]
prob_offset = Vector{Float64}[]
αs = LinRange(0,2π,51)[1:end-1]        # movement directions over which to sweep
offsets = LinRange(-2r,2r,31)
x₀s_offset = eachrow(x₀_opt' .+ [cos(α_opt+π/2) sin(α_opt+π/2)] .* offsets) # offsets over which to sweep

x₀s_rotated = [[cos(α+π)*r+0.5*grid_params.xscale,sin(α+π)*r+0.5*grid_params.yscale] for α ∈ αs] # path starting points
for θ ∈ [8,4]
    objects[:n].θ_syn = θ
    objects[:seg1].θ_syn = θ
    objects[:seg2].θ_syn = θ

    counts = zeros(Int, length(vs))
    @showprogress 1.0 "Sweeping run-speeds ..." for (i,v) ∈ enumerate(vs)
        path = generate_straight_path((path_trange[1],path_trange[1]+2r/v), α_opt, v, x₀_opt)
        for t ∈ 1:trials
            spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter, background_rate=λ_background)

            if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
                counts[i] += 1
            end
        end
    end
    push!(prob_speed, counts ./ trials)


    ## Vary the orientation
    counts = zeros(Int, length(αs))
    @showprogress 1.0 "Sweeping directions ..." for (i,(α,x₀)) ∈ enumerate(zip(αs, x₀s_rotated))
        path = generate_straight_path(path_trange, α, v_opt, x₀)
        for t ∈ 1:trials
            spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter, background_rate=λ_background)

            if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
                counts[i] += 1
            end
        end
    end
    push!(prob_rotated, counts ./ trials)

    ## Vary the shift
    counts = zeros(Int, length(offsets))
    @showprogress 1.0 "Sweeping offsets ..." for (i,x₀) ∈ enumerate(x₀s_offset)
        path = generate_straight_path(path_trange, α_opt, v_opt, x₀)
        for t ∈ 1:trials
            spikes, logger = run_path!(net, path_trange, path, populations, λ; t_jitter, background_rate=λ_background)

            if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
                counts[i] += 1
            end
        end
    end
    push!(prob_offset, counts ./ trials)
end

## Plotting
# Set up plots
show_spines = (
    :leftspinevisible => true,
    :rightspinevisible => true,
    :bottomspinevisible => true,
    :topspinevisible => true
)

yticks,ytickformat = make_manual_ticks(collect(0:-5:-60), vcat([""],["$(grp)$(sub)" for grp in ["A","B","C"] for sub in ["₅ ", "₁₀","₁₅","₂₀"]]))

fig = Figure(resolution = (0.75textwidth, 0.75textwidth))
ax11 = fig[1,1] = Axis(fig; title="a.    Effective paths", titlealign=:left, backgroundcolor=:transparent, xlabel="x coordinate [mm]", ylabel="y coordinate [mm]", 
    show_spines...)
# hidedecorations!(ax11)
gl = fig[1,2:3] = GridLayout()
ax12 = gl[1,1] = Axis(fig; 
    title="b.    Activity for the highlighted path", titlealign=:left, xlabel="time [ms]", 
    yticks=([-50, -30, -10], ["C","B","A"]), 
    yminorticks=Makie.IntervalsBetween(20, true),
    yminorticksvisible = true, yminorgridvisible = true, 
)
ax13 = gl[1,2] = Axis(fig; aspect=DataAspect(), backgroundcolor=:transparent)
hidedecorations!(ax13)
ax21 = fig[2,1] = Axis(fig; title="c.    Optimal path", titlealign=:left, backgroundcolor=:transparent, ylabel="y coordinate", show_spines...)
hidexdecorations!(ax21)
ax22 = fig[2,2] = Axis(fig; title="d.    Rotated paths", titlealign=:left, backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax22)
ax23 = fig[2,3] = Axis(fig; title="e.    Translated paths", titlealign=:left, backgroundcolor=:transparent, show_spines...)
hidedecorations!(ax23)
ax31 = fig[3,1] = Axis(fig, xlabel="run speed [mm/ms]", ylabel="spike probability", )
ax32 = fig[3,2] = Axis(fig, xlabel="angle [deg]")
ax33 = fig[3,3] = Axis(fig, xlabel="offset [mm]")

rowsize!(fig.layout, 1, Aspect(1, grid_params.yscale/grid_params.xscale))
rowsize!(fig.layout, 2, Aspect(1, grid_params.yscale/grid_params.xscale))
colsize!(gl, 2, Relative(0.1))
linkaxes!(ax11, ax21)
linkaxes!(ax11, ax23)
linkyaxes!(ax31, ax32)
linkyaxes!(ax31, ax33)



# Plot the stochastically sampled paths
for path in paths[2:end]
    lines!(ax11, 1000 .*(path.x.-domain[2][1]/2), 1000 .*(path.y.-domain[2][2]/2), linewidth=2, color=RGBAf0(0.2,0.2,0.2,0.5))
    arrows!(ax11, 1000 .* [Point2f0(path.x[end],path.y[end]).-domain[2]./2], [Point2f0(path.x[end] - path.x[end-1], path.y[end] - path.y[end-1])], linewidth=2, color=RGBAf0(0.2,0.2,0.2,0.5))
end
lines!(ax11, 1000 .*(paths[1].x.-domain[2][1]/2), 1000 .*(paths[1].y.-domain[2][2]/2), linewidth=4, color=color_4)
arrows!(ax11, 1000 .* [Point2f0(paths[1].x[end],paths[1].y[end]).-domain[2]./2], [Point2f0(paths[1].x[end] - paths[1].x[end-1], paths[1].y[end] - paths[1].y[end-1])], linewidth=4, color=color_4)


# Plot the spikes, plateaus and rectangles to highlight plateau triggers
seg1 = get_trace(:seg1, special_path.logger.data)
statetrace!(ax12, [0;1000 .* seg1.t;200], [DPC.voltage_low;seg1.state;DPC.voltage_low], Dict(DPC.voltage_low=>:transparent, DPC.voltage_elevated=>color_1_25, DPC.voltage_high=>color_1_50), -20.05, 19.9)
seg2 = get_trace(:seg2, special_path.logger.data)
statetrace!(ax12, [0;1000 .* seg2.t;200], [DPC.voltage_low;seg2.state;DPC.voltage_low], Dict(DPC.voltage_low=>:transparent, DPC.voltage_elevated=>color_2_25, DPC.voltage_high=>color_2_50), -40.05, 19.9)
for (i,color) in enumerate((color_1, color_2, color_3))
    for j in 1:20
        epsp = get_trace(Symbol("syn$(i)$(lpad(j,2,"0"))"), special_path.logger.data)
        statetrace!(ax12, [0;1000 .* epsp.t;200], [0;Int.(epsp.state);0], Dict(0=>:transparent, 1=>color), -(i-1)*20 .+ -j, 0.9)
    end
end
linesegments!(ax12, 1000 .* repeat(plateau_starts_1.t, inner=2), repeat([-20, 0], outer=length(plateau_starts_1.t)), color=color_1, linewidth=2)
for t in plateau_extended_1.t
    lines!(ax12, 1000 .* [t,t], [-20.25, -0.25], color=color_1, linewidth=2)
end
linesegments!(ax12, 1000 .* repeat(plateau_starts_2.t, inner=2), repeat([-40, -20], outer=length(plateau_starts_2.t)), color=color_2, linewidth=2)
for t in plateau_extended_2.t
    lines!(ax12, 1000 .* [t,t], [-40.25, -20.25], color=color_2, linewidth=2)
end
linesegments!(ax12, 1000 .* repeat(spike_times.t, inner=2), repeat([-60, -40], outer=length(spike_times.t)), color=color_3, linewidth=2)

xlims!(ax12, 1000 .* path_trange.+[-2,6])
# Plot the neuron
plot!(ax13, objects[:n],
    branch_width=1, 
    branch_length=8.0, 
    color=Dict(:n=>color_3, :seg1=>color_1, :seg2=>color_2)
)

# Plot the optimal path
path_opt = generate_straight_path(path_trange, α_opt, v_opt, x₀_opt)
(path_start,path_end) = path_opt.(path_trange)
arrows!(ax21, 1000 .* Point2f0[path_start .- domain[2]./2], 1000 .* Point2f0[path_end .- path_start], linewidth=2, color=:black)


# Plot the speed-dependent spike probability
vlines!(ax31, [v_opt], linestyle=:dash, color=:gray, linewidth=2)
for (prob, linestyle) in zip(prob_speed, (:solid, :dash))
    lines!(ax31, vs, prob; color=color_4, linewidth=2, linestyle)
end

# Plot the rotated paths
pos = Point2f0[]
dir = Point2f0[]
for (α,x₀) in zip(αs, x₀s_rotated)
    path = generate_straight_path(path_trange, α, v_opt, x₀)
    (path_start,path_end) = path.(path_trange)
    push!(pos, path_start .- domain[2]./2)
    push!(dir, path_end .- path_start)
end
arrows!(ax22, 1000 .* pos, 1000 .* dir, linewidth=2, color=RGBAf0.(0.2,0.2,0.2,prob_rotated[1]))

# Plot the orientation-dependent spike probability
vlines!(ax32, [α_opt].*180/π, linestyle=:dash, color=:gray, linewidth=2)
for (prob, linestyle) in zip(prob_rotated, (:solid, :dash))
    lines!(ax32, αs.*180/π, prob; color=color_4, linewidth=2, linestyle)
end



# Plot the offset paths
pos = Point2f0[]
dir = Point2f0[]
for (p,x₀) in zip(prob_offset[1], x₀s_offset)
    path = generate_straight_path(path_trange, α_opt, v_opt, x₀)
    (path_start,path_end) = path.(path_trange)
    push!(pos, path_start .- domain[2]./2)
    push!(dir, path_end .- path_start)
end
arrows!(ax23, 1000 .* pos, 1000 .* dir, linewidth=2, color=RGBAf0.(0.2,0.2,0.2,prob_offset[1]))

# Plot the offset-dependent spike probability
vlines!(ax33, [0], linestyle=:dash, color=:gray, linewidth=2)
for (prob, linestyle) in zip(prob_offset, (:solid, :dash))
    lines!(ax33, 1000 .*offsets, prob; color=color_4, linewidth=2, linestyle)
end

xlims!(ax11, -500 * domain[2][1], 500 * domain[2][1])
ylims!(ax11, -500 * domain[2][2], 500 * domain[2][2])
ylims!(ax12, -60.5, 0.5)

save(joinpath("figures","place_cells.pdf"), fig)
save(joinpath("figures","place_cells.svg"), fig)
save(joinpath("figures","place_cells.png"), fig)
fig
