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