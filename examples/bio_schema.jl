using DPC, CairoMakie
include("utils.jl")

config = """
neurons:
  - id: neuron
    branches:
      - id: S
        branches:
          - id: S1
            branches:
              - id: S11
                branches:
                  - id: S111
                  - id: S112
              - id: S12
                branches:
                  - id: S121
                  - id: S122
                  - id: S123
                  - id: S124
          - id: S2
            branches:
              - id: S21
                branches:
                  - id: S211
                  - id: S212
                  - id: S213
                  - id: S214
                  - id: S215
                  - id: S216
              - id: S22
                branches:
                  - id: S221
                  - id: S222
                  - id: S223
"""

(net,objects) = load_network(YAML_source=config)

fig = Figure(resolution=(600,600),aspect=DataAspect())
ax = fig[1,1] = Axis(fig, backgroundcolor=:transparent)

diams = get_branch_diameters(objects[:neuron], root_diam=0.2, ratio_fun=(d,n)->d*0.75)
lengths = DefaultDict(0.75, :S=>2.0, :S1=>1.75, :S2=>1.25, :S11=>1.0, :S12=>1.0, :S21=>1.0, :S22=>1.0)
α = 0/180*π
angles = DefaultDict(20/180*π, :S=>α, :S1=>α, :S2=>α, :S11=>α, :S12=>α, :S21=>α, :S22=>α)

cols = cgrad(:viridis, 6, categorical=true, rev=true)
dists = get_root_distances(objects[:neuron])
colors = Dict(k=>cols[2+v] for (k,v) in pairs(dists))
plot!(ax, objects[:neuron], angle_between=angles, branch_taper=0.4, branch_width=diams, branch_length=lengths, color=colors, padding = (0.0, 0.0))

hidedecorations!(ax)
hidespines!(ax)
xlims!(ax, (-2.5,3.5))
ylims!(ax, (-0.5,6.5))

save(joinpath(@__DIR__, "figures", "bio_schema.svg"), fig)
save(joinpath(@__DIR__, "figures", "bio_schema.png"), fig)
save(joinpath(@__DIR__, "figures", "bio_schema.pdf"), fig)
fig
