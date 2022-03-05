using DPC, CairoMakie
include("utils.jl")
# set_theme!(presentation_theme)
set_theme!(mytheme)

config0 = """
neurons:
  - id: C
    θ_syn: 0
    branches:
      - id: A
        θ_syn: 5
"""

config1 = """
neurons:
  - id: C
    θ_syn: 0
    θ_seg: 1
    branches:
      - id: A
        θ_syn: 5
        branches:
          - id: B
            θ_syn: 5
"""

config2 = """
neurons:
  - id: C
    θ_syn: 0
    θ_seg: 1
    branches:
      - id: A
        θ_syn: 5
      - id: B
        θ_syn: 5
"""

num_synapses = 10
prob_A = fill(0.0, num_synapses)
prob_B = fill(0.0, num_synapses)
volleys = Dict(:A => (prob_A, ones(num_synapses)), :B => (prob_B, ones(num_synapses)))

probabilities = 0:0.01:1

firing_probabilities0 = Vector{Vector{Float64}}(undef, num_synapses)
(net0,objects0) = load_network(YAML_source=config0)
for θ in 1:num_synapses
    println("Computing for single segment, θ=$(θ)")
    objects0[:A].θ_syn = θ
    firing_probabilities0[θ] = [(prob_A .= a; P_fire(objects0[:C], volleys)) for a ∈ probabilities]
end

println("Computing for AND neuron")
(net1,objects1) = load_network(YAML_source=config1)
firing_probabilities1 = [(prob_A .= a; prob_B .= b; P_fire(objects1[:C], volleys)) for a ∈ probabilities, b ∈ probabilities]

println("Computing for OR neuron")
(net2,objects2) = load_network(YAML_source=config2)
firing_probabilities2 = [(prob_A .= a; prob_B .= b; P_fire(objects2[:C], volleys)) for a ∈ probabilities, b ∈ probabilities]

## Plotting

fig = Figure(resolution=(0.75textwidth,0.7textwidth))
ax1a = fig[1,1] = Axis(fig, backgroundcolor=:transparent,
    title="a.    single\n       segment", titlealign=:left, tellwidth=false    
)

ax2a = fig[1,2] = Axis(fig, backgroundcolor=:transparent,
    title="b.    sequential\n       segments", titlealign=:left, tellwidth=false
)

ax3a = fig[1,3] = Axis(fig, backgroundcolor=:transparent,
    title="c.    parallel\n       segments", titlealign=:left, tellwidth=false
)

ax1b = fig[2,1] = Axis(fig, xlabel="p₁", ylabel="Plateau probability")#, aspect=DataAspect()
ax2b = fig[2,2] = Axis(fig, xlabel="p₁", ylabel="p₂", )
ax3b = fig[2,3] = Axis(fig, xlabel="p₁", ylabel="p₂", )
cell_col_1 = fig[3,1]
cell_col_2 = fig[3,2:3]

colors = cgrad(:viridis,length(firing_probabilities0), categorical=true)
for (prob,col) in zip(firing_probabilities0, colors)
    lines!(ax1b, probabilities, prob, color=col, linewidth=3)
end
ax_col_1 = Colorbar(cell_col_1, height=10, colormap=colors, limits=(0.5,10.5), ticks=1:10, label="TS₁", vertical=false, flipaxis = false)

contourf!(ax2b, probabilities, probabilities, firing_probabilities1, colormap=:viridis, colorrange=(0,1))

contourf!(ax3b, probabilities, probabilities, firing_probabilities2, colormap=:viridis, colorrange=(0,1))

p_n=plot!(ax3a, objects2[:C], ports=Dict(:C=>[],:A=>[:A, :dummy],:B=>[:dummy, :B]), angle_between=20/180*π, branch_width=0.2, branch_length=1.0, color=Dict(:C=>RGBAf(0.5,0.5,0.5,0.5), :A=>color_1, :B=>color_2))
ports = Dict(p_n.attributes[:ports][])
arrows!(ax3a, [ports[:A]-Point2f0(0.6,0),ports[:B]-Point2f0(0.45,0)], [Point2f0(0.55,0),Point2f0(0.4,0)], linewidth=2, arrowsize = [10,10])
text!.(ax3a, "A~B(10,P₁)", position=ports[:A]-Point2f0(0.6,0), align=(:left, :top), textsize=14, color=:black)
text!.(ax3a, "B~B(10,P₂)", position=ports[:B]-Point2f0(0.45,0), align=(:left, :bottom), textsize=14, color=:black)

p_n=plot!(ax2a, objects1[:C], ports=Dict(:C=>[],:A=>[:A],:B=>[:B]), angle_between=20/180*π, branch_width=0.2, branch_length=1.0, color=Dict(:C=>RGBAf(0.5,0.5,0.5,0.5), :A=>color_1, :B=>color_2))
ports = Dict(p_n.attributes[:ports][])
arrows!(ax2a, [ports[:A]-Point2f0(0.35,0),ports[:B]-Point2f0(0.35,0)] , Observable([Point2f0(0.3,0),Point2f0(0.3,0)]), linewidth=2, arrowsize = [10,10])
text!.(ax2a, "A~B(10,P₁)", position=ports[:A]-Point2f0(0.35,0), align=(:left, :bottom), textsize=14, color=:black)
text!.(ax2a, "B~B(10,P₂)", position=ports[:B]-Point2f0(0.35,0), align=(:left, :bottom), textsize=14, color=:black)

p_n=plot!(ax1a, objects0[:C], ports=Dict(:C=>[:C],:A=>[:A]), angle_between=20/180*π, branch_width=0.2, branch_length=1.0, color=Dict(:C=>RGBAf(0.5,0.5,0.5,0.5), :A=>color_1, :B=>color_2))
ports = Dict(p_n.attributes[:ports][])
arrows!(ax1a, [ports[:A]-Point2f0(0.35,0)], [Point2f0(0.3,0)], linewidth=2, arrowsize = 10)
text!.(ax1a, "A~B(10,P₁)", position=ports[:A]-Point2f0(0.35,0), align=(:left, :bottom), textsize=14, color=:black)


ax_col_2 = Colorbar(cell_col_2, height=10, colormap=:viridis, limits=(0,1), label="Plateau probability for TS₁=TS₂=5", vertical=false, flipaxis = false)

hidedecorations!(ax3a)
hidespines!(ax3a)
hidedecorations!(ax2a)
hidespines!(ax2a)
hidedecorations!(ax1a)
hidespines!(ax1a)

hideydecorations!(ax3b)
colgap!(fig.layout, 1, Fixed(45)) 
colgap!(fig.layout, 2, Fixed(45)) 
rowsize!(fig.layout, 1, Aspect(1, 1.5))
rowsize!(fig.layout, 2, Aspect(1,1)) 

ylims!(ax1a, -0.25,2.25)
ylims!(ax2a, -0.25,2.25)
ylims!(ax3a, -0.25,2.25)
xlims!(ax1a, -0.8333,0.8333)
xlims!(ax2a, -0.8333,0.8333)
xlims!(ax3a, -0.8333,0.8333)

xlims!(ax1b, 0,1)
xlims!(ax2b, 0,1)
xlims!(ax3b, 0,1)
ylims!(ax1b, 0,1)
ylims!(ax2b, 0,1)
ylims!(ax3b, 0,1)
ax2b.yticks[] = [0,0.5,1.0]



# save(joinpath("figures","probabilistic.pdf"), fig)
save(joinpath("figures","probabilistic.svg"), fig)
# save(joinpath("figures","probabilistic.png"), fig)
fig
