using ADSP, DifferentialEquations, DataStructures, Plots, ProgressMeter
include("utils.jl")

# Setup
name = :unidirectional
cfg = ("examples/grid_cells/cfg/unidirectional.yaml")
segment_colors = Dict(
    :soma=>gridcell_colors[2],
    :d1=>gridcell_colors[1],
    :d2=>gridcell_colors[3]
)

trange = (0.0, 0.15)                                         # time duration of each simulation run
n = 100
αs = LinRange(0,2π,n+1)[1:end-1]
r = 1.5*grid_params.r
v = 2r/(trange[2]-trange[1])
x₀s_rotated = [[cos(α+π)*r+0.5*grid_params.xscale,sin(α+π)*r+0.5*grid_params.yscale] for α ∈ αs]
trials = 100
α_opt = 2π/6
x₀_opt = [cos(α_opt+π)*r+0.5*grid_params.xscale,sin(α_opt+π)*r+0.5*grid_params.yscale]
offsets = LinRange(-2r,2r,31)
x₀s_offset = eachrow(x₀_opt' .+ [cos(α_opt+π/2) sin(α_opt+π/2)] .* offsets)

# Load neuron
logged_segments = OrderedDict(segment=>PropertyID(SegmentID(NeuronID(:readout), segment), :active) for segment ∈ keys(segment_colors)) 


#======== Run the simulations =========#

# Vary the orientation
counts = zeros(Int, length(αs))
@showprogress 1.0 "Sweeping directions ..." for (i,(α,x₀)) ∈ enumerate(zip(αs, x₀s_rotated))
    p = generateLinePath(trange, α, v, x₀)
    for t ∈ 1:trials
        s,logger = run_path(cfg, p, logged_segments; group_size=20, background_rate=10)
        if any(logger.soma)
            counts[i] += 1
        end
    end
end
prob_rotated = counts ./ trials
hist_rotated = (hcat(cos.(αs),sin.(αs)) .*r .* prob_rotated) .+ 0.5 .* [grid_params.xscale grid_params.yscale]

# Vary the shift
counts = zeros(Int, length(offsets))
@showprogress 1.0 "Sweeping offsets ..." for (i,x₀) ∈ enumerate(x₀s_offset)
    p = generateLinePath(trange, α_opt, v, x₀)
    for t ∈ 1:trials
        s,logger = run_path(cfg, p, logged_segments; group_size=20, background_rate=10)
        if any(logger.soma)
            counts[i] += 1
        end
    end
end
prob_offset = counts ./ trials
hist_offset = [cos(α_opt+π/2) sin(α_opt+π/2)].*offsets .+ [cos(α_opt) sin(α_opt)] .* prob_offset .*r .+ 0.5 .* [grid_params.xscale grid_params.yscale]

  
#======== Generate the plots =========#

plt_rotated=begin
    plt = plot(legend=false, grid=false, xlims=(domain[1][1],domain[2][1]), ylims=(domain[1][2],domain[2][2]), xticks=false, yticks=false, framestyle=:box, aspect_ratio=1, title="direction sensitivity")
    for (c,α,x₀) ∈ collect(zip(prob_rotated,αs,x₀s_rotated))#[1:5:end]
        p = generateLinePath(trange, α, v, x₀)
        (p_start,p_end) = p.(trange)
        plot!([p_start[1],p_end[1]],[p_start[2],p_end[2]], arrow=:head, linewidth=2, color=:gray, alpha=c)
        # plot!([α,α],[-r,r])
    end

    plot!(hist_rotated[:,1], hist_rotated[:,2], linewidth=2, m=4, color=:red)
    plot!(x->sin(x)*r+0.5*grid_params.xscale,x->cos(x)*r+0.5*grid_params.yscale, 0,2π, linecolor=:gray, linewidth=1, st=:shape, fill=nothing)
    plot!(x->sin(x)*r*0.5+0.5*grid_params.xscale,x->cos(x)*r*0.5+0.5*grid_params.yscale, 0,2π, linecolor=:gray, linewidth=1, st=:shape, fill=nothing,annotate=[(0.5*grid_params.xscale+r, 0.5*grid_params.yscale,Plots.text("100% ", :right, :gray)),(0.5*grid_params.xscale+0.5r, 0.5*grid_params.yscale, Plots.text("50% ",:right, :gray))])

    p = generateLinePath(trange, α_opt, v, x₀_opt)
    (p_start,p_end) = p.(trange)
    plot!([p_start[1],p_end[1]],[p_start[2],p_end[2]], arrow=:head, color=:black, linestyle=:dot, linewidth=5)
    plt
end

plt_offset=begin
    plt = plot(legend=false, grid=false, xlims=(domain[1][1],domain[2][1]), ylims=(domain[1][2],domain[2][2]), xticks=false, yticks=false, framestyle=:box, aspect_ratio=1, title="location sensitivity")
    for (c,x₀) ∈ collect(zip(prob_offset,x₀s_offset))
        p = generateLinePath(trange, α_opt, v, x₀)
        (p_start,p_end) = p.(trange)
        plot!([p_start[1],p_end[1]],[p_start[2],p_end[2]], arrow=:head, linewidth=2, color=:gray, alpha=c)
    end

    pp = repeat([cos(α_opt+π/2) sin(α_opt+π/2)], 1,3).* [-2r,2r] .+ 0.5 .* repeat([grid_params.xscale grid_params.yscale],1,3) .+ repeat([cos(α_opt) sin(α_opt)], 1, 3) .* [0 0 0.5r 0.5r r r]
    plot!(pp[:,1:2:end], pp[:,2:2:end], color=:silver, arrow=:both, linewidth=2,annotate=[(([0.65 0.35]*pp[:,1:2])...,Plots.text("0% ", :top, :gray, rotation=α_opt*360/2π-90)),(([0.65 0.35]*pp[:,3:4])...,Plots.text("50% ", :top, :gray, rotation=α_opt*360/2π-90)),(([0.65 0.35]*pp[:,5:6])..., Plots.text("100% ", :top, :gray, rotation=α_opt*360/2π-90))])
    plot!(hist_offset[:,1], hist_offset[:,2], linewidth=2, m=4, color=:red, seriestype=:shape, fillopacity=0.25)
    plot!([0.3 0.7; 0.7 0.3]*pp[:,1], [0.3 0.7; 0.7 0.3]*pp[:,2], color=:red, arrow=:both, linewidth=5)

    p = generateLinePath(trange, α_opt, v, x₀_opt)
    (p_start,p_end) = p.(trange)
    plot!([p_start[1],p_end[1]],[p_start[2],p_end[2]], arrow=:head, color=:black, linestyle=:dot, linewidth=5)
    plt
end


plt_path = plt_rfs = plt_spikes = plt_plateaus = nothing

plt_path=plot()

prog = Progress(num_paths+1, 1, "Sampling paths...")
i=0
ProgressMeter.update!(prog, i)
while i < num_paths
    # generate a single path and the response
    p = generatePath(trange, path_params, generate_u₀(path_params, domain))
    s,logger = run_path(cfg, p, logged_segments; group_size=20, background_rate=10)

    # if this path ended in a somatic plateau -> generate the plot
    if any(logger[!,:soma])
        global i+=1
        plot!(plt_path, t->p(t)[1], t->p(t)[2], trange..., legend=false, grid=false, xlims=(domain[1][1],domain[2][1]), ylims=(domain[1][2],domain[2][2]), xticks=false, yticks=false, framestyle=:box, aspect_ratio=1, linewidth=1, color="#cccccc80", arrow=:arrow)
    end
    ProgressMeter.update!(prog, i)
end


while true
    # generate a single path and the response
    local p = generatePath(trange, path_params, generate_u₀(path_params, domain))
    s,logger = run_path(cfg, p, logged_segments; group_size=20, background_rate=10)

    # if this path led to somatic plateau, generate the plot
    if any(logger[!,:soma])
        # plot the path itself
        plot!(plt_path, t->p(t)[1], t->p(t)[2], trange..., legend=false, grid=false, xlims=(domain[1][1],domain[2][1]), ylims=(domain[1][2],domain[2][2]), xticks=false, yticks=false, framestyle=:box, aspect_ratio=1, linewidth=3, color=:black, arrow=:arrow, title="effective trajectories")

        # plot each receptive field population response
        global plt_rfs = plot(legend=false,xlabel="time [s]", ylabel="corr.", title="receptive field responses", yticks=[0,0.5,1.0], ylims=[0,1])
        for (j,rf) ∈ enumerate(rfs)
            plot!(rf∘p, trange..., linewidth=2, color=gridcell_colors[j])
        end
        
        # plot the resulting spike trains
        global plt_spikes = plot_spike_raster(trange, s, 5e-3; colors=gridcell_colors, title="population spike trains", ylabel="neuron")

        # plot the triggered plateaus on the dendritic segments
        global plt_plateaus = plot(xlabel="time [s]", title="dendritic plateaus", ylabel="segment")
        for (i,segment) ∈ enumerate(keys(logged_segments))
            plot!(logger[!,:t], logger[!,segment].+i,   seriestype=:steppre, legend=false, yticks=false, color=segment_colors[segment], linewidth=2, xlims=trange, fillrange=i)
        end
        ProgressMeter.update!(prog, num_paths+1)
        break
    end
end

l = @layout [
    a{0.5w, 0.5h} c{0.5h}
    b{0.5w, 0.5h} [d{0.6h}
     e{0.2h}
     f{0.2h}
    ]
]

# compose the subplots
plot(plt_rotated, plt_offset, plt_path, plt_spikes, plt_rfs, plt_plateaus, layout=l, size=(700,800))
savefig("examples/grid_cells/figures/$(name)_summary.svg")

# println("Generating paths...")
# plot()
# savefig("examples/grid_cells/figures/$(name)_sample_paths.svg")
