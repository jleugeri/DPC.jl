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
max_N = 100
p_opt = fill(1.0, max_N)
θ_opt = fill(10, max_N)

## Compute mutual information for all N from 1 to 100
@progress "Computing MI..." for N = 1:max_N
    II .= I.(reshape(pp,(:,1)),reshape(θθ,(1,:)),Ref(P_X),N)
    p_idx,θ_idx = argmax(II).I
    θ_opt[N] = θθ[θ_idx]
    p_opt[N] = pp[p_idx]
end

## Plot figure
fig = Figure(resolution=(0.5latex_textwidth,0.5latex_textwidth))
h = nothing

ax1 = fig[1,1] = Axis(fig; title="A    single segment", xlabel="spike volley size (X)", ylabel="plateaus (N)", yticks=[0,1])
ax2 = fig[1,2] = Axis(fig; title="B    $(max_N) segments", xlabel="spike volley size (X)")



linkxaxes!(ax1,ax2)

xx = collect(1:x_max)
stairs!(ax1, xx, [p_z_x(1,x,p_opt[1],θ_opt[1],P_X) for x in xx], color=color_1, linewidth=2,step=:center)
#vlines!(ax1, [θ_opt[1]], linestyle=:dash)


N = max_N
nn = collect(0:N)
P = [p_n_x(n,x,p_opt[N],θ_opt[N],P_X,N) for x in xx, n in nn]
E_n = [N .* p_z_x(1,x,p_opt[N],θ_opt[N],P_X) for x in xx]

hm=heatmap!(ax2, xx[θ_opt[N]:end], nn, P[θ_opt[N]:end,:], colormap=(RGBAf0(0.9,0.9,0.9,1.0),RGBAf0(0.0,0.0,0.0,1.0)), colorrange=(0,0.2))
translate!(hm,0,0,-100)
lines!(ax2, xx, E_n, linewidth=2, color=color_2)
#vlines!(ax2, [θ_opt[N]], linestyle=:dash)

#fig[1,3] = Colorbar(fig, hm, label=L"P(N|X)", size=5)

ax3 = fig[2,1:end] = Axis(fig; title="C    Optimal stochasticity", ylabel=L"opt. \theta", xlabel="number of segments", xscale=log10, yscale=log10, yaxisposition=:right, ylabelcolor=:gray, yticklabelcolor=:gray, yticks=[4,5,6,7,8,9,10,11], xticks=([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100],["1","2","3","","","","","","","10","20","30","","","","","","","100"]))
stairs!(ax3, 1:max_N, θ_opt, color=:gray, step=:center)

fn(x) = (x-4)/(11-4) * (1.0-p_opt[N])+p_opt[N]

ax4 = fig[2,1:end] = Axis(fig; xscale=log10, yscale=log10, yticks=fn.([4,5,6,7,8,9,10,11]), ylabel=L"P_{syn}",ytickformat="{:.2f}")
stairs!(ax4, 1:max_N, p_opt, color=:black, step=:center)

scatter!(ax4,[1,N],[p_opt[1],p_opt[N]],markersize=8,color=[color_1,color_2])

#lines!(ax4, 1:max_N, 1 ./ sqrt.(2 .* (1:max_N)), color=:black, linestyle=:dash)

hidexdecorations!(ax4)

linkxaxes!(ax3,ax4)

fig
##
## save result
save(joinpath(@__DIR__, "figures","dithering.png"), fig)
save(joinpath(@__DIR__, "figures","dithering.svg"), fig)
save(joinpath(@__DIR__, "figures","dithering.pdf"), fig)
