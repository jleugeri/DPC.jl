using LinearAlgebra, Printf, SparseArrays, Random, Distributions, CairoMakie
include("utils.jl")

## Define experimental parameters ##############################################

N = 1000            # dimensionality of the input space
ρ = 0.2             # sparsity
#p_back = 0.01       # background firing probability (noise!)
θ = 180             # firing threshold of the segments
num_locations = 20  # number of latitudes & longitudes on the test-point grid 
num_samples = 1000  # number of samples to draw per location

# arbitrarily choosen directions of interest
A_dir,B_dir,C_dir = [0.1, 0.2, 0.5],[0.2, 0.5, 0.2],[1.0, 0.3, 0.3]

θ_α = θ_β = θ_γ = θ
latitudes = LinRange(0, pi/2, num_locations)     # only upper hemisphere
longitudes = LinRange(0, pi/2, num_locations)    # only first sector

## generate basis vectors and gramian

# X,Y and Z are the N-dimensional basis vectors for the three features
M = SparseArrays.sprand(Bool, N, 3, ρ)
X,Y,Z = eachcol(M)

# compute gramian matrix for the basis vectors
gramian = Symmetric(M'*M)

# the basis vectors are not normalized
scales = [1.0/sqrt(gramian[i,i]) for i in 1:3]
gramian_n = scales' .* gramian .* scales

normalize_with_gramian(v;G=gramian_n) = v/√(v'*G*v)

A,B,C = normalize_with_gramian.((A_dir, B_dir, C_dir))

# compute the (normalized!) coefficients for the weight vectors
# each weight vector (for one segment) corresponds to one point of interest
# takes into account relative length of the basis vectors
# α,β,γ = [normalize(x .* scales) for x in (A,B,C)]
α,β,γ = A,B,C 

# compute the actual synaptic transmission probabilities:
# an input spike due to feature X should be transmitted with probability α₁ (or β₁ or γ₁)
# an input spike due to feature Y should be transmitted with probability α₂ (or β₂ or γ₂)
# an input spike due to feature Z should be transmitted with probability α₃ (or β₃ or γ₃)
# => the total transmission probability for the input is the OR of these probabilities
p_transmit = 1.0 .- 
    (1.0 .- (X .* [α[1] β[1] γ[1]])) .* 
    (1.0 .- (Y .* [α[2] β[2] γ[2]])) .* 
    (1.0 .- (Z .* [α[3] β[3] γ[3]]))
w_α,w_β,w_γ = eachcol(p_transmit)

## Experiment 1: draw grid of testpoints & compute responses ###################

# draw different normalized coefficient vectors for positive combinations of X, Y and Z,
# i.e. a latitude/longitude grid on the surface of the positive sector of the unit sphere
grid_x,grid_y,grid_z = compute_lat_long_grid(latitudes,longitudes)

# for each coefficient vector, compute the probability of a spike in each input:
# an input spike due to feature X occurs with probability grid_x
# an input spike due to feature Y occurs with probability grid_y
# an input spike due to feature Z occurs with probability grid_z
# => the total probability for the input to spike is the OR of these probabilities
p_spike_grid = 1.0 .- 
    (1.0 .- (X .* prepend_dim(grid_x))) .* 
    (1.0 .- (Y .* prepend_dim(grid_y))) .* 
    (1.0 .- (Z .* prepend_dim(grid_z)))

# sample 'num_samples' random input spike vectors for each grid point
# final shape of 'samples_grid': N × num_locations × num_locations × num_samples
samples_grid = zeros(Bool, size(p_spike_grid)..., num_samples)
for id in CartesianIndices(p_spike_grid)
    samples_grid[id, :] = rand(Bernoulli(p_spike_grid[id]), num_samples)
end

# see what input would be transmitted for each segment with what probability
# final shape of 'transmission_probability': 3 × num_locations × num_locations
plateau_probability_grid = zeros(Float64, 3, num_locations, num_locations)
for (segment,(w,θ)) in enumerate(((w_α,θ_α),(w_β,θ_β),(w_γ,θ_γ)))
    # draw which of the input spikes are transmitted by the stochastic synapses
    transmitted_spikes = rand.(Bernoulli.(w .* samples_grid))
    
    # compute which of the transmitted spike-volleys crossed the segment's threshold
    triggered_plateau = sum(transmitted_spikes, dims=1) .> θ

    # compute the plateau probability
    plateau_probability_grid[segment,:,:] = mean(triggered_plateau, dims=4)
end

## Experiment 2: compute responses for the target pattern ######################

# for each target pattern, compute the probability of a spike in each input:
p_spike_target = 1.0 .- 
    (1.0 .- (X .* [A[1] B[1] C[1]])) .* 
    (1.0 .- (Y .* [A[2] B[2] C[2]])) .* 
    (1.0 .- (Z .* [A[3] B[3] C[3]]))

# sample 'num_samples' random input spike vectors for each target
# final shape of 'samples_target': N × 3 × num_samples
samples_target = zeros(Bool, size(p_spike_target)..., num_samples)
for id in CartesianIndices(p_spike_target)
    samples_target[id, :] = rand(Bernoulli(p_spike_target[id]), num_samples)
end

# see what input would be transmitted for each segment with what probability
# final shape of 'transmission_probability': 3
plateau_probability_target = zeros(Float64, 3)
for (segment,(w,θ)) in enumerate(((w_α,θ_α),(w_β,θ_β),(w_γ,θ_γ)))
    # draw which of the input spikes are transmitted by the stochastic synapses
    transmitted_spikes = rand.(Bernoulli.(w .* samples_target[:,segment,:]))
    
    # compute which of the transmitted spike-volleys crossed the segment's threshold
    triggered_plateau = sum(transmitted_spikes, dims=1) .> θ

    # compute the plateau probability
    plateau_probability_target[segment] = mean(triggered_plateau)
end

# compute detection probability: each target must have been detected
detection_probability = prod(plateau_probability_target)

## Compute 2D projections ######################################################

# X_trans,Y_trans and Z_trans are the 2D projection of the three feature's basis vectors
X_trans,Y_trans,Z_trans = compute_projection(gramian_n)

"""
    project_2D(v;X_trans=X_trans,Y_trans=Y_trans,Z_trans=Z_trans, G=gramian_n, check_valid=true)

Computes a projetion into the 2D space spanned by the basis `X_trans`,`Y_trans`,`Z_trans` with Gramian `G`.
"""
function project_2D(v; X_trans=X_trans,Y_trans=Y_trans,Z_trans=Z_trans, G=gramian_n, check_valid=true)
    @assert v'*gramian_n*v ≈ 1.0 "Coefficient vector must be normalized! (v has norm '$(√(v'*gramian_n*v))')"
    
    # compute the dot-products with the (normalized) basis vectors
    dp = G*v
    
    # compute the greater-circle distance -> distance in 2D
    (d₁,d₂,d₃)=acos.(dp)
    
    # compute the triangulated position in 2D
    return triangulate((d₁,d₂,d₃),X_trans,Y_trans,Z_trans; check_valid)
end

# A_trans,B_trans and C_trans are the 2D projection of the three target patterns
A_trans,B_trans,C_trans = project_2D.((A,B,C))


## Draw figure #################################################################
fig = Figure()

# draw overview axis
ax_left = fig[1:3,1] = Axis(fig, aspect = DataAspect())

xlims!(ax_left, min(X_trans[1],Y_trans[1],Z_trans[1])-0.5,max(X_trans[1],Y_trans[1],Z_trans[1])+0.5,)
ylims!(ax_left, min(X_trans[2],Y_trans[2],Z_trans[2])-0.5,max(X_trans[2],Y_trans[2],Z_trans[2])+0.5,)
hidedecorations!(ax_left)
hidespines!(ax_left)

cut_to_sector = true
cut_to_triangle = !cut_to_sector
# draw coordinate grid
for (i,r1) ∈ enumerate(reverse(cos.(0:0.1:acos(maximum(gramian_n[[2,3,6]])))))
    #    v₁ = (1.0 - v₂*v₂*G₂₂ - v₂*v₃*G₂₃ - v₃*v₂*G₃₂-v₃v₃G₃₃)/(c+v₂G₂₁+v₃G₃₁)
    for j in 1:3
        println(r1," ",j)
        x,y = get_trimetric_contour(j, r1, -0.05:0.25:1.5,-0.05:0.25:1.5, gramian_n, (X_trans, Y_trans, Z_trans); iterations=4, cut_to_sector, cut_to_triangle)
        lines!(ax_left, x, y, color=:silver, linewidth=1)
    end

    #  # compute actual dot-product
    #  (tick,_,dp) = project_2D([r1,0,rest],X_trans,Y_trans,Z_trans)
    #      # draw r1 on the left
    #  dx = tick-project_2D([r1,√(0.01)*rest,√(1-0.01)*rest],X_trans,Y_trans,Z_trans).pt
    #  if dx[1]≈0.0
    #      angle = -π/8
    #      dx = Point2(-1.0,0.5)
    #  else
    #      angle = atan(dx[2]/dx[1])
    #      dx /= sqrt(dx[1]^2+dx[2]^2)
    #  end
    #  annotations!(ax, [@sprintf("%.2f",dp[1])],[tick+0.02*dx], align=(:right, :center), rotation=angle)
end

if cut_to_sector
    r1=project_2D.(get_geodesic([1,0,0],[0,1,0],gramian_n)[2:end-1])
    r2=project_2D.(get_geodesic([0,1,0],[0,0,1],gramian_n)[2:end-1])
    r3=project_2D.(get_geodesic([0,0,1],[1,0,0],gramian_n)[2:end-1])
    lines!(ax_left, Point2[[X_trans];r1;[Y_trans];r2;[Z_trans];r3;[X_trans]], color=:black, linewidth=3)
else
end

# # draw frame
# for j in 1:3
#     x,y = get_isometric_contour(j,maximum(gramian_n[j,1:3 .!=j]),0.0:0.5:1.5,-0.25:0.5:1.25; iterations=6)
#     lines!(ax_left, x, y, color=:black, linewidth=2)
# end

# draw the corner points
scatter!(ax_left, [X_trans,Y_trans,Z_trans], color=[:black, :silver, :silver])

# draw the points of interest
scatter!(ax_left, [A_trans,B_trans,C_trans], color=[:red,:green,:blue], markersize=20)
fig
##
annotations!([@sprintf("%s(%.2f,%.2f,%.2f)",name,p.dp...) for (name,p) in zip(("A","B","C"),(A_trans,B_trans,C_trans))], [A_trans.pt,B_trans.pt,C_trans.pt], align=[(:center,:bottom), (:center,:top), (:center,:top)], offset=[Point2(-40.0,10.0),Point2(0.0,-10.0),Point2(0.0,-10.0)])

arrows!(ax, [A_trans.pt,B_trans.pt] +  0.05.* [B_trans.pt-A_trans.pt, C_trans.pt-B_trans.pt], 0.875 .* [B_trans.pt-A_trans.pt, C_trans.pt-B_trans.pt], linewidth=3)

colsize!(fig.layout, 1, Relative(0.7))
fig

for (case,(w,pt)) in enumerate(zip(eachcol(p_transmit),(A_trans,B_trans,C_trans)))
    ax_i = fig[case,2] = Axis(fig, aspect=DataAspect())
    hidedecorations!(ax_i)
    hidespines!(ax_i)
    transmitted_spikes = dropdims(sum(rand.(Bernoulli.(w .* samples)), dims=1),dims=1)
    plateau = transmitted_spikes .> θ
    plateau_prob = dropdims(mean(plateau,dims=3),dims=3)

    #heatmap!(ax_i, plateau_prob)
    mesh!(ax_i, vertices, faces, color=plateau_prob, colorrange =(0,1))
    scatter!(ax_i, [pt.pt], color=:red)
end

Colorbar(fig[4, 2], limits = (0, 1), colormap = :viridis, vertical = false)

fig
##





vertices = vec([transform([xv, yv, zv].*scales , X_trans, Y_trans, Z_trans).pt for (xv, yv, zv) in zip(x2, y2, z2)])
faces = [((i-1)*num_locations + j.+(k==1 ? [0 num_locations 1] : [num_locations num_locations+1 1]) for i in  1:num_locations-1 for j in 1:num_locations-1 for k in 1:2)...;]


fn(x)=all(x.> 0)
fn(x)=all((gramian_n*x).>0)

scatter(Point3.(p), color = ifelse.(isnan.(first.(p_p)), ifelse.(fn.(p),:orange,:red) , ifelse.(fn.(p),:green,:blue)), markersize=20)
