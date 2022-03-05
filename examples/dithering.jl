using Distributions, CairoMakie, ProgressLogging
include("utils.jl")

# Probabilities of the individual detector
"""
p(z|x) = P(Y-θ ≥ 0) = P(Y≥θ) = 1-F_Y(θ;p,x)
"""
function p_z_x(z,x,p,θ,P_X)
    r = cdf(Binomial(x,p),θ-1)
    z==0 ? r : 1.0-r
end

"""
p(x,z) = p(x)*p(z|x)
"""
p_zx(z,x,p,θ,P_X) = pdf(P_X,x) *p_z_x(z,x,p,θ,P_X)

"""
p(z) = ∑ₓp(x)*p(z|x)
"""
p_z(z,p,θ,P_X) = sum(χ -> p_zx(z,χ,p,θ,P_X), P_X.a:P_X.b)

# Probabilities for multiple detectors
"""
p(n|x) = P(ζ=n), ζ~Binomial(N,p(z=1|x))
"""
p_n_x(n,x,p,θ,P_X,N) = pdf(Binomial(N,p_z_x(1,x,p,θ,P_X)),n)

"""
p(n,x) = p(x)p(n|x)
"""
p_nx(n,x,p,θ,P_X,N) = pdf(P_X,x) *p_n_x(n,x,p,θ,P_X,N)

"""
p(n) = ∑ₓp(x)*p(n|x)
"""
p_n(n,p,θ,P_X,N) = sum(χ -> p_nx(n,χ,p,θ,P_X,N), P_X.a:P_X.b)

"""
I(Z;X)=∑ₓ∑ₙ p(x)*p(n|x)*(log(p(n|x))-log(p(n)))

Special case for N=1:
p(n=1|x) = p(z=1|x), p(n=1)=p(z=1) ⇒
I(Z;X)=∑∑ p(x)*p(z|x)*(log(p(z|x))-log(p(z)))
"""
function I(p,θ,P_X,N)
    args=(p,θ,P_X,N)
    I = 0.0
    for n ∈ 0:N
        q_n = p_n(n,args...)
        if q_n ≈ 0
            continue
        end
        for x ∈ P_X.a:P_X.b
            q_n_x = p_n_x(n,x,args...)
            q_nx = p_nx(n,x,args...)
            if q_n_x ≈ 0
                continue
            end
            I += q_nx * log2( q_n_x / q_n )
        end
    end
    I
end

## Set up parameters

x_max = 20
P_X = DiscreteUniform(1,x_max)
pp = collect(LinRange(0,1,250))
θθ = collect(1:x_max)
II = zeros(Float64, length(pp),length(θθ))
p_opt = Float64[]
θ_opt = Int[]

## Compute mutual information for all N from 1 to 100
@progress "Computing MI..." for N = 1:100
    II .= I.(reshape(pp,(:,1)),reshape(θθ,(1,:)),Ref(P_X),N)
    p_idx,θ_idx = argmax(II).I
    push!(θ_opt, θθ[θ_idx])
    push!(p_opt, pp[p_idx])
end

## Plot figure

N = 25
fig = Figure(resolution=(0.75textwidth,0.25textwidth))

for (i,N) in enumerate((1,25))
    # plot heatmap
    ax = fig[1,2i-1] = Axis(fig, titlealign=:left, title="$(N) detectors", xlabel="Transmission probability", ylabel="Threshold")
    II .= I.(reshape(pp,(:,1)),reshape(θθ,(1,:)),Ref(P_X),N)
    h = heatmap!(ax, pp, θθ, II, colormap = :heat)
    Colorbar(fig[1,2i], h, label = "Mutual Information I(N;X) in bits")

    # plot 
    lines!(ax, p_opt, θ_opt, color=:black)
    c = fill(:black,length(p_opt))
    s = fill(5,length(p_opt))
    c[N] = :red
    s[N] = 10
    scatter!(ax, p_opt, θ_opt, markersize=s, color=c)
    
end
colgap!(fig.layout, 2, 100)

display(fig)

## save result
save(joinpath("figures","dithering.png"), fig)
save(joinpath("figures","dithering.svg"), fig)
save(joinpath("figures","dithering.pdf"), fig)
