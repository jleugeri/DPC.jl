using LinearAlgebra, GLMakie
include("utils.jl")

pt1 = normalize([1,0.2,0.1])
pt2 = normalize([0.1,1,0.1])
pt3 = normalize([0.1,0.1,1])

proj1 = (x₁,y₁) = [0,0]
proj2 = (x₂,y₂) = [1,0]
proj3 = (x₃,y₃) = [0.5,√(1.0-0.5^2)]


##
fig=Figure()
ax=fig[1,1]=Axis(fig)
scatter!(ax,Point2[proj1,proj2,proj3])

X = normalize([0.01,0.4,0.5])  # normalize(randn(3))

d₁,d₂,d₃ = acos.(X)

x1,y1=implicit_function_contour((x,y)->((x-proj1[1])^2+(y-proj1[2])^2)*d₃^2-d₁^2*((x-proj3[1])^2+(y-proj3[2])^2),0:0.5:4,-2:0.5:2; constraint=(x,y)->all([x,y] .!= 0.0))

x2,y2=implicit_function_contour((x,y)->((x-proj1[1])^2+(y-proj1[2])^2)*d₂^2-d₁^2*((x-proj2[1])^2+(y-proj2[2])^2),0:0.5:4,-2:0.5:2; constraint=(x,y)->all([x,y] .!= 0.0))

x3,y3=implicit_function_contour((x,y)->((x-proj2[1])^2+(y-proj2[2])^2)*d₃^2-d₂^2*((x-proj3[1])^2+(y-proj3[2])^2),0:0.5:4,-2:0.1:2; constraint=(x,y)->all([x,y] .!= 0.0),iterations=5)

# take points proj1 and proj2

lines!(ax, x1,y1)
lines!(ax, x2,y2)
lines!(ax, x3,y3)

# theory says this should be circles
r₂ = d₁*d₂*norm(proj2-proj1)/(d₁^2-d₂^2)
r₃ = d₁*d₃*norm(proj3-proj1)/(d₁^2-d₃^2)
lines!(ax, Circle(Point2(proj2[1]+d₂/d₁*r₂,0),r₂), linestyle=:dash, color=:red)
   
fig
##

G₁₂,G₁₃,G₂₃ = G[1,2],G[1,3],G[2,3]
f(v₂,v₃) = v₂^2 * (G₁₂^2 - 1.0) + 2.0 * v₂*v₃*(G₁₂*G₁₃-G₂₃) + v₃^2 * (G₁₃^2 - 1.0) + 1.0-c^2
g(v₂,v₃) = (1.0 - v₂^2 - 2v₂*v₃*G₂₃ -v₃^2)/(c+v₂*G₁₂+v₃*G₁₃)

function constraint(v₂,v₃)
    v = [g(v₂,v₃),v₂,v₃]
    return (v'*G*v ≈ 1.0) && trimetric_constraint(acos.(clamp.(G*v,-1,1)),(X_trans),(Y_trans),(Z_trans))
end
v₂,v₃ = implicit_function_contour(f,0.0:0.5:1.5,-0.25:0.5:1.25; constraint)
scatter([g.(v₂,v₃);;v₂;;v₃])
##


println("Constraint: ",-norm(proj3 - proj2)*d₁^2/(d₁^2-d₃^2) + norm(proj1 - proj2)*d₁*d₂/(d₁^2-d₂^2) + norm(proj1 - proj3)*(d₁*d₃)/(d₁^2-d₃^2))
