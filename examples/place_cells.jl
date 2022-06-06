using DPC, CairoMakie, ProgressMeter, Distributions

#include(joinpath(@__DIR__, "utils.jl"))
include("utils.jl")
include(joinpath(@__DIR__, "place_cells", "utils.jl"))

## Initialize the network
(net,objects) = load_network(joinpath(@__DIR__, "place_cells", "network.yaml"), weight_type=BernoulliSynapseWeight{Float64})

num_inputs_per_class = 20

# Set up the place-cell populations:
populations = [
    (
        # the receptive field functions are Gaussians centered at the gridcell_centers with radii determined by gridcell_radii
        rf = v-> ismissing(v) ? 0.0 : exp(-sum((v.-μ).^2)/(2*(grid_params.r/3)^2)),
        neurons = n
    ) for (μ,n) in zip(gridcell_centers,[[objects[Symbol("i$(i)$(lpad(j,2,"0"))")] for j in 1:20] for i in 1:3])
]


## Randomly sample stochastic paths 
num_paths = 10
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
            global special_path = path
        end
    end
    ProgressMeter.update!(prog, i)
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

fig = Figure(resolution = (0.75latex_textwidth, 0.6latex_textwidth))
ax11 = fig[1,1] = Axis(fig; title="A    Effective paths", titlealign=:left, backgroundcolor=:transparent, xlabel="x coordinate [mm]", ylabel="y coordinate [mm]", 
    show_spines...)
hidedecorations!(ax11)

ax12 = fig[1,2] = Axis(fig; title="B    Sequence of volleys", titlealign=:left)
hidedecorations!(ax12)

ax13 = fig[1,3] = Axis(fig; aspect=DataAspect(), backgroundcolor=:transparent)
hidedecorations!(ax13)

gl2 = fig[2,1:3] = GridLayout()
ax21 = gl2[1,1] = Axis(fig; title="C    Same path on different time-scales", titlealign=:center, backgroundcolor=:transparent)
ax22 = gl2[2,1] = Axis(fig; ylabel="slow", backgroundcolor=:transparent)
ax23 = gl2[3,1] = Axis(fig; backgroundcolor=:transparent)

gl3 = fig[3,1:3] = GridLayout()
ax31 = gl3[1,1] = Axis(fig; backgroundcolor=:transparent)
ax32 = gl3[2,1] = Axis(fig; ylabel="fast", backgroundcolor=:transparent)
ax33 = gl3[3,1] = Axis(fig; backgroundcolor=:transparent)

gl4 = fig[4,1:3] = GridLayout()
ax41 = gl4[1,1] = Axis(fig; backgroundcolor=:transparent)
ax42 = gl4[2,1] = Axis(fig; ylabel="replay", backgroundcolor=:transparent)
ax43 = gl4[3,1] = Axis(fig; backgroundcolor=:transparent, xaxisalign=:right)
spike_axes = [ax21 ax22 ax23 ; ax31 ax32 ax33 ; ax41 ax42 ax43]

colsize!(fig.layout, 1, Relative(0.3))
colsize!(fig.layout, 3, Relative(0.1))
rowsize!(fig.layout, 1, Aspect(1, grid_params.yscale/grid_params.xscale))
hideydecorations!.((ax21,ax22,ax23,ax31,ax32,ax33,ax41,ax42,ax43); label=false)
hidexdecorations!.((ax21,ax22,ax23,ax31,ax32,ax33,ax41,ax42); grid=false)

rowgap!(fig.layout,2,Relative(0.01))
rowgap!(fig.layout,3,Relative(0.01))

rowgap!(gl2,0)
rowgap!(gl3,0)
rowgap!(gl4,0)

## Plot the stochastically sampled paths
for path in paths[2:end]
    lines!(ax11, 1000 .*(path.x.-domain[2][1]/2), 1000 .*(path.y.-domain[2][2]/2), linewidth=2, color=RGBAf0(0.6,0.6,0.6,1))
    arrows!(ax11, 1000 .* [Point2f0(path.x[end],path.y[end]).-domain[2]./2], [Point2f0(path.x[end] - path.x[end-1], path.y[end] - path.y[end-1])], linewidth=2, color=RGBAf0(0.6,0.6,0.6,1))
end
lines!(ax11, 1000 .*(paths[1].x.-domain[2][1]/2), 1000 .*(paths[1].y.-domain[2][2]/2), linewidth=4, color=color_4)
arrows!(ax11, 1000 .* [Point2f0(paths[1].x[end],paths[1].y[end]).-domain[2]./2], [Point2f0(paths[1].x[end] - paths[1].x[end-1], paths[1].y[end] - paths[1].y[end-1])], linewidth=4, color=color_4)

fig

## Plot the spike responses for different speeds

for (i,duration) in enumerate((0.2,0.1,0.02))

    # create one sample for each run speed
    path(t) = special_path((t+duration)/duration*(path_trange[2]-path_trange[1])+path_trange[1])
    timeout = true
    for j in 1:1000
        global spikes, logger = run_path!(net, (-duration,0.0), path, populations, λ; t_jitter, background_rate=λ_background)
        if any(x->(x.object == :n && x.event == :spikes), eachrow(logger.data))
            timeout=false
            break
        end
    end
    if timeout
        error("failed to generate a path with spike for duration $(duration)!")
    end
    spike_times = filter(x->(x.object==:n && x.event==:spikes), logger.data)

    first_spike = minimum(spike_times.t)

    # plot fake axis
    for j in 1:3
        # plot the background
        poly!(spike_axes[i,j], Rect2D((-1000.0 * (duration+first_spike),-0.5),(1000.0 * duration,1.0)), color=RGBf0(0.9,0.9,0.9))
    end

    # plot spike probabilities
    tt = LinRange(-duration-first_spike,0.0-first_spike,250)
    for (j,population) in enumerate(populations)

        c = [color_1_25,color_2_25,color_3_25][j]
        rf_j = population.rf.(path.(tt .+ first_spike))
        band!(spike_axes[i,j], 1000.0 .* tt, rf_j./2,-rf_j./2, color=c)
    end

    # extract input population spikes
    s_t = [Float64[],Float64[],Float64[]]
    s_n = [Float64[],Float64[],Float64[]]
    for ev in spikes
        s=string(ev.target.id)
        population = parse(Int,s[2])
        neuron = parse(Int,s[3:end])
        push!(s_t[population],ev.t .- first_spike)
        push!(s_n[population],(neuron-1)/(num_inputs_per_class-1)-0.5)
    end
    
    
    
    # plot the actual spike volleys
    for j in 1:3
        c = [color_1,color_2,color_3][j]
        linesegments!(spike_axes[i,j], 
            1000.0 .* repeat(s_t[j], inner=2), 
            [s_n[j]'; s_n[j]'.+ 1/(num_inputs_per_class-1)][:], 
            color=c, 
            linewidth=2
        )
    end


    # extract neuron-internal events
    plateau_starts_1 = filter(x->(x.object==:seg1 && x.event==:plateau_starts), logger.data)
    plateau_starts_2 = filter(x->(x.object==:seg2 && x.event==:plateau_starts), logger.data)
    plateau_extended_1 = filter(x->(x.object==:seg1 && x.event==:plateau_extended), logger.data)
    plateau_extended_2 = filter(x->(x.object==:seg2 && x.event==:plateau_extended), logger.data)

    for (j,p) in enumerate((plateau_starts_1,plateau_starts_2,spike_times))
        annotations!(spike_axes[i,j], [["A","B","C"][j]], [Point2f0(1000.0 * minimum(p.t .- first_spike),0.0)]; align=(:right,:center), offset=(-10,0), textsize=14)
        vlines!(spike_axes[i,j], [0.0], color=:black, linestyle=:dot)
        vlines!(spike_axes[i,j], 1000.0 .* (p.t .- first_spike), color=:black, linewidth=2)
    end

    for (j,p) in enumerate((plateau_extended_1,plateau_extended_2))
        if (!isempty(p.t)) 
            vlines!(spike_axes[i,j], 1000.0 .* (p.t .- first_spike), color=RGBAf0(0.0,0.0,0.0,0.05), linewidth=2)
        end
    end

end

xlims!(ax43,-0.2,0.05)
linkxaxes!.(Ref(ax21), (ax22,ax23,ax31,ax32,ax33,ax41,ax42,ax43))
ax43.bottomspinevisible[]=true
ax43.xlabel[]= "spike-aligned time [ms]"

fig

## Plot the neuron
plot!(ax13, objects[:n],
    branch_width=1, 
    branch_length=8.0, 
    color=Dict(:n=>color_3, :seg1=>color_1, :seg2=>color_2)
)

fig
##

save(joinpath(@__DIR__, "figures","place_cells.pdf"), fig)
save(joinpath(@__DIR__, "figures","place_cells.svg"), fig)
save(joinpath(@__DIR__, "figures","place_cells.png"), fig)
fig
