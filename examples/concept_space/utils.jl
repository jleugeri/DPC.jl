using CairoMakie, Polynomials
import MDBM


"""
compute_projection(M)

Computes the 2D projection of the three basis vectors from the normalized gramian M.

If `check_valid==true`, asserts that the computed point satisfies the desired distances.
"""
function compute_projection(M; check_valid=true)
    @assert all(i->M[i,i]≈1.0, 1:size(M,1)) "Gramian must be normalized!"

    ## generate the transformed points for plotting
    # how different are X,Y and Z to each other? -> distance in the plot
    # d₁: X<->Y, d₂: X<->Z, d₃: Y <-> Z 
    d₁,d₂,d₃ = acos.((M[1,2], M[1,3], M[2,3]))

    ###############################################
    # Compute third projection point (dx,dy):
    # Constraints 
    #   (I): dx²+dy² = d₂²
    #  (II): (d₁-dx)²+dy² = d₃²
    #
    # Combine III=I-II:
    # (III): dx = (d₁²+d₂²-d₃²)/(2d₁)
    #
    # Using dx, solve I for dy
    ###############################################

    dx = (d₁^2+d₂^2-d₃^2)/(2d₁)
    dy = √(d₂^2-dx^2)
    # verify that dx and dy satisfy the constraints

    if check_valid
        @assert dx^2+dy^2≈d₂^2 "Distance d₂ not satisfied: $(√(dx^2+dy^2)) ≠ $(d₂)"
        @assert (d₁-dx)^2+dy^2≈d₃^2 "Distance d₃ not satisfied: $(√((d₁-dx)^2+dy^2)) ≠ $(d₃)"
    end

    return Point2(0,0),Point2(d₁,0),Point2(dx,dy)
end


"""
    prepend_dim(x)

Prepends a singleton dimension to the shape of `x`
"""
prepend_dim(x) = reshape(x, (1,size(x)...))

"""
    compute_lat_long_grid(latitudes,longitudes)

Computes a grid on the surface of the unit sphere at the given `latitudes` and `longitudes`
"""
function compute_lat_long_grid(latitudes,longitudes)
    x = [cos(φv) * sin(θv) for θv in latitudes, φv in longitudes]
    y = [sin(φv) * sin(θv) for θv in latitudes, φv in longitudes]
    z = [cos(θv) for θv in latitudes, φv in 2 .* longitudes]
    return (x,y,z)
end

"""
    implicit_function_contour(f, x_range, y_range, iterations=5)

Computes the contour of the given function `f(x,y)==0` in the range given by `x_range` and `y_range`.
Uses the MDBM algorithm with given number of `iterations`.
Returns vectors `x` and `y` with the coordinates of the contour.
"""
function implicit_function_contour(f, x_range, y_range; iterations=5, constraint=(x,y)->1)
    prob=MDBM.MDBM_Problem(f, [MDBM.Axis(x_range,"x"),MDBM.Axis(y_range,"y")]; constraint)
    MDBM.solve!(prob,iterations)
    queue = [collect(x) for x in MDBM.connect(prob)]

    while length(queue)>1
        delete_q = zeros(Bool,length(queue))
        for i in eachindex(queue)
            for (j,chain2) in enumerate(queue[1:i-1])
                if delete_q[j]
                    continue
                end
                if first(chain2)==last(chain2)
                    delete_q[j] = true
                elseif first(queue[i]) == first(chain2)
                    queue[i]=[reverse(chain2)[1:end-1];queue[i]]
                    delete_q[j] = true
                elseif first(queue[i]) == last(chain2)
                    queue[i]=[chain2[1:end-1];queue[i]]
                    delete_q[j] = true
                elseif last(queue[i]) == first(chain2)
                    queue[i]=[queue[i][1:end-1];chain2]
                    delete_q[j] = true
                elseif last(queue[i]) == last(chain2)
                    queue[i]=[queue[i][1:end-1];reverse(chain2)]
                    delete_q[j] = true
                end
            end
        end
        if !any(delete_q)
            break
        end
        deleteat!(queue, delete_q)
    end
    x = Float64[]
    y = Float64[]
    for q in queue
        qx,qy=MDBM.getinterpolatedsolution(prob.ncubes[q],prob)
        push!(x, qx..., NaN)
        push!(y, qy..., NaN)
    end

    if !isempty(x) && (isnan(first(x)) || isnan(first(y)))
        popfirst!(x)
        popfirst!(y)
    end

    if !isempty(x) && (isnan(last(x)) || isnan(last(y)))
        pop!(x)
        pop!(y)
    end

    return (;x,y)
end

"""
    triangulate((d₁,d₂,d₃),(x₁,y₁),(x₂,y₂),(x₃,y₃); check_valid=true)

Takes three distances and the three 2D positions.
Returns a 2D project of `v`.

If `check_valid==true`, asserts that the computed point satisfies the desired distances.
"""
function triangulate((d₁,d₂,d₃),(x₁,y₁),(x₂,y₂),(x₃,y₃); check_valid=true)::Tuple{Float64,Float64}
    if any(isnan,(d₁,d₂,d₃))
        return (NaN,NaN)
    end

    @assert d₁≥0 && d₂≥0 && d₃≥0 "Cannot have negative difference!"

    #===========================================================================
    Compute position of point (x,y) from distances d₁,d₂ and d₃:
    Constraints: the relative distances to the projection points must be preserved
    - from the ratio d₁/d₃: ||P₁-X||²d₃² = ||P₃-X||²d₁²  
                <=> ((x₁-x)²+(y₁-y)²)d₃² = ((x₃-y)²+(y₃-y)²)d₁²
    - from the ratio d₂/d₃: ||P₂-X||²d₃² = ||P₃-X||²d₂²  
                <=> ((x₂-x)²+(y₂-y)²)d₃² = ((x₃-x)²+(y₃-y)²)d₂²

    Expansion:
        (I): 0 = x²(d₃²-d₁²) + 2x(x₃d₁²-x₁d₃²) + 2y(y₃d₁²-y₁d₃²) + y²(d₃²-d₁²) + [x₁²d₃²-x₃²d₁²+y₁²d₃²-y₃²d₁²]
               = x²α₁ + 2xβ₁ + 2yγ₁ + y²α₁ + δ₁
       (II): 0 = x²(d₃²-d₂²) + 2x(x₃d₂²-x₂d₃²) + 2y(y₃d₂²-y₂d₃²) + y²(d₃²-d₂²) + [x₂²d₃²-x₃²d₂²+y₂²d₃²-y₃²d₂²]
               = x²α₂ + 2xβ₂ + 2yγ₂ + y²α₂ + δ₂
    where    α₁= (d₃²-d₁²), α₂= (d₃²-d₂²), β₁= (x₃d₁²-x₁d₃²), β₂= (x₃d₂²-x₂d₃²), γ₁= (y₃d₁²-y₁d₃²), γ₂= (y₃d₂²-y₂d₃²)
             δ₁= [x₁²d₃²-x₃²d₁²+y₁²d₃²-y₃²d₁²], δ₂= [x₂²d₃²-x₃²d₂²+y₂²d₃²-y₃²d₂²]


    CASE DISTINCTION
    ================

    CASE 1: α₁=α₂=0
    --------------
    α₁=α₂=0 ⇒ d₁²=d₂²=d₃²=d ⇒ 
              β₁= d²(x₃-x₁), β₂= d²(x₃-x₂), 
              γ₁= d²(y₃-y₁), γ₂= d²(y₃-y₂), 
              δ₁= [x₁²-x₃²+y₁²-y₃²]d², 
              δ₂= [x₂²-x₃²+y₂²-y₃²]d²
    
        (I¹): x = -yγ₁/β₁ - δ₁/(2β₁)
       (II¹): x = -yγ₂/β₂ - δ₂/(2β₂)
      (III¹): 0 = y(γ₁/β₁-γ₂/β₂) + δ₁/(2β₁) - δ₂/(2β₂)
                = y((y₃-y₁)/(x₃-x₁)-(y₃-y₂)/(x₃-x₂)) + [x₁²-x₃²+y₁²-y₃²]/(2(x₃-x₁)) - [x₂²-x₃²+y₂²-y₃²]/(2(x₃-x₂))
                = 2y((y₃-y₁)(x₃-x₂)-(y₃-y₂)(x₃-x₁)) + (x₁-x₃)(x₃-x₂)(x₁-x₂) + (y₁-y₃)(y₁+y₃)(x₃-x₂) - (y₂-y₃)(y₂+y₃)(x₃-x₁)
                = 2y(x₁(y₃-y₂)+(y₁-y₃)x₂+(y₂-y₁)x₃) + (x₁²+y₁²)(x₃-x₂) + (x₂²+y₂²)(x₁-x₃) + (x₃²+y₃²)(x₂-x₁)
    =>        y = [(x₁²+y₁²)(x₃-x₂) + (x₂²+y₂²)(x₁-x₃) + (x₃²+y₃²)(x₂-x₁)]/[2(x₁(y₃-y₂)+(y₁-y₃)x₂+(y₂-y₁)x₃)]
    =>        x = [(x₁²+y₁²)(y₂-y₃) + (x₂²+y₂²)(y₃-y₁) + (x₃²+y₃²)(y₁-y₂)]/[2(x₁(y₃-y₂)+(y₁-y₃)x₂+(y₂-y₁)x₃)]
    (Unsurprisingly, this is the circum-center!)


    CASE 2: α₁=0, α₂≠0
    ------------------
        (I²): x = -yγ₁/β₁ - δ₁/(2β₁)
    Plug back into (II):
       (II²): 0 = x²α₂ + 2xβ₂ + 2yγ₂ + y²α₂ + δ₂
                = α₂y²γ₁² + y²α₂β₁² + α₂yγ₁δ₁ - 2yγ₁β₁β₂ + 2yγ₂β₁² + α₂δ₁²/4 - δ₁β₁β₂ + δ₂β₁²
                = y²[α₂γ₁² + α₂β₁²] + y[α₂γ₁δ₁ -2γ₁β₁β₂ + 2γ₂β₁²] + [α₂δ₁²/4 - δ₁β₁β₂ + δ₂β₁²]

    Solve for y.
    Solve I² for x.

    CASE 3: α₂=0, α₁≠0
    ------------------
       (II²): x = -yγ₂/β₂ - δ₂/(2β₂)
    Plug back into (I):
        (I²): 0 = x²α₁ + 2xβ₁ + 2yγ₁ + y²α₁ + δ₁
                = α₁y²γ₂² + y²α₁β₂² +α₁yγ₂δ₂ - 2yγ₂β₁β₂ + 2yγ₁β₂² + α₁δ₂²/4 - δ₂β₁β₂ + δ₁β₂²
                = y²[α₁γ₂² + α₁β₂²] + y[α₁γ₂δ₂ -2γ₂β₁β₂ + 2γ₁β₂²] + [α₁δ₂²/4 - δ₂β₁β₂ + δ₁β₂²]

    Solve for y.
    Solve II² for x.

    CASE 4: α₁≠0, α₂≠0
    ------------------
    Eliminate x² and y² by setting III = I·α₂-II·α₁ and solve for x:
      (III³): 0 = 2x[β₁α₂-β₂α₁] + 2y[γ₁α₂-γ₂α₁] + [δ₁α₂-δ₂α₁] 
                = 2xβ₃ + 2yγ₃ + δ₃ 
        <=>   x = -yγ₃/β₃ - δ₃/(2β₃)
    where  β₃ = β₁α₂-β₂α₁, γ₃ = γ₁α₂-γ₂α₁, δ₃ = δ₁α₂-δ₂α₁

    Plug back into (II):
    (II⁴): 0 = y²[γ₃²α₁ + α₁β₃²] + y[γ₃δ₃α₁ - 2β₁γ₃β₃ + 2γ₁β₃²] + [α₁δ₃²/4 - β₁δ₃β₃ + δ₁β₃²]

    Solve for y.
    Solve III for x.
    ================

    =#

    CASE = 0
    (x,y) = if d₁≈0.0 # CASE 0: corner 1
        CASE=0
        (x₁,y₁)
    elseif d₂≈0.0 # CASE 0: corner 2
        CASE=0
        (x₂,y₂)
    elseif d₃≈0.0 # CASE 0: corner 3
        CASE=0
        (x₃,y₃)
    elseif d₃≈d₂≈d₁ # CASE 1: special case circum-center
        CASE=1
        D = (2*((y₂-y₃)*x₁+(y₃-y₁)*x₂+(y₁-y₂)*x₃))
        y = ((x₁^2+y₁^2)*(x₃-x₂) + (x₂^2+y₂^2)*(x₁-x₃) + (x₃^2+y₃^2)*(x₂-x₁))/D
        x = ((x₁^2+y₁^2)*(y₂-y₃) + (x₂^2+y₂^2)*(y₃-y₁) + (x₃^2+y₃^2)*(y₁-y₂))/D
        (x,y)
    else        # general case: solve polynomial!
        α₁= d₃^2-d₁^2
        α₂= d₃^2-d₂^2
        β₁= x₃*d₁^2-x₁*d₃^2
        β₂= x₃*d₂^2-x₂*d₃^2
        γ₁= y₃*d₁^2-y₁*d₃^2
        γ₂= y₃*d₂^2-y₂*d₃^2
        δ₁= x₁^2*d₃^2-x₃^2*d₁^2+y₁^2*d₃^2-y₃^2*d₁^2
        δ₂= x₂^2*d₃^2-x₃^2*d₂^2+y₂^2*d₃^2-y₃^2*d₂^2

        c₂=c₁=c₀=0.0
        if α₁ ≈ 0.0  # CASE 2
            CASE=2
            c₂ = α₂*γ₁^2 + α₂*β₁^2
            c₁ = α₂*γ₁*δ₁ -2*γ₁*β₁*β₂ + 2*γ₂*β₁^2
            c₀ = α₂*δ₁^2/4 - δ₁*β₁*β₂ + δ₂*β₁^2
        elseif α₂ ≈ 0.0 # CASE 3 
            CASE=3
            c₂ = α₁*γ₂^2 + α₁*β₂^2
            c₁ = α₁*γ₂*δ₂ -2*γ₂*β₁*β₂ + 2*γ₁*β₂^2
            c₀ = α₁*δ₂^2/4 - δ₂*β₁*β₂ + δ₁*β₂^2
        else         # CASE 4
            CASE=4
            β₃ = β₁*α₂-β₂*α₁
            γ₃ = γ₁*α₂-γ₂*α₁
            δ₃ = δ₁*α₂-δ₂*α₁

            c₂ = α₁*γ₃^2 + α₁*β₃^2
            c₁ = α₁*γ₃*δ₃ - 2*β₁*β₃*γ₃ + 2*β₃^2*γ₁
            c₀ = α₁*δ₃^2/4 - β₁*β₃*δ₃ + β₃^2*δ₁
        end

        r = filter(isreal,roots(Polynomial([c₀,c₁,c₂])))
        if isempty(r)
            @info ("Could not find a solution for y-coordinate for distances $((d₁,d₂,d₃))")
            return (NaN,NaN)
        end
        
        candidate = (NaN,NaN)
        candidate_dist = Inf
        for y in r
            x = 0
            if CASE==2
                x = -y*γ₁/β₁ - δ₁/(2β₁)
            elseif CASE==3
                x = -y*γ₂/β₂ - δ₂/(2β₂)
            else
                x = -y*γ₃/β₃ - δ₃/(2β₃)
            end

            candidate_new = (x,y)
            candidate_new_dist = √(
                  ((√(x₁-x)^2+(y₁-y)^2)-d₁)^2 # deviation from d₁
                + ((√(x₂-x)^2+(y₂-y)^2)-d₂)^2 # deviation from d₂
                + ((√(x₃-x)^2+(y₃-y)^2)-d₃)^2 # deviation from d₃
            )
            if candidate_new_dist ≤ candidate_dist
                candidate_dist = candidate_new_dist
                candidate = candidate_new
            end
        end
        
        if length(r) > 1
            #@info ("Found multiple real candidate solutions (y∈$(r)), picked (x,y) = $(candidate).")
        end
        
        candidate
    end
    return (x,y)
end


function get_geodesic(P₁,P₂,G;num_points=500)
    P₁,P₂ = normalize_with_gramian.((P₁,P₂);G)
    u = normalize_with_gramian(cross(P₁, P₂);G)
    max_angle = acos(P₁'*G*P₂)
    R1 = [0.0 -u[3] u[2]; u[3] 0.0 -u[1]; -u[2] u[1] 0.0]
    R2 = u*u'

    [normalize_with_gramian(cos(θ)*P₁+sin(θ)*R1*P₁+(1.0-cos(θ))*R2*P₁;G) for θ in LinRange(0.0,max_angle,num_points)]
    # #rotation axis:      u = P₁×P₂
    # #rotation direction: d = x×u = x×(P₁×P₂) = P₁*(x'*G*P₂)-P₂*(x'*G*P₁)
    # x = P₁
    # res = Vector{Vector{Float64}}(undef, num_points)
    # for i in 1:num_points
    #     x -= 0.01 * P₁*(x'*G*P₂)-P₂*(x'*G*P₁)
    #     x = normalize_with_gramian(x;G)
    #     res[i] = x
    # end
    # return res
end


function trimetric_constraint((v₁,v₂,v₃),Q₁,Q₂,Q₃,G)
    D = acos.(G*[v₁,v₂,v₃])
    conditions = zeros(Float64,3)     
    for i in 0:2
        P₁,P₂,P₃ = circshift([Q₁,Q₂,Q₃],i)
        d₁,d₂,d₃ = circshift(D,i)
    
        old_D₁₂ = norm(P₁-P₂)
        old_D₁₃ = norm(P₁-P₃)
        old_D₂₃ = norm(P₂-P₃)
        r₂ = old_D₁₂*d₁*d₂/(d₁^2-d₂^2)
        r₃ = old_D₁₃*d₁*d₃/(d₁^2-d₃^2)
        α₂ = 1+d₂^2/(d₁^2-d₂^2)
        α₃ = 1+d₃^2/(d₁^2-d₃^2)
        new_D₂₃ = sqrt(old_D₁₃^2*α₃^2 + old_D₁₂^2*α₂^2 - α₃*α₂*(old_D₁₃^2+old_D₁₂^2-old_D₂₃^2))
        conditions[i+1] = r₂ + r₃ - new_D₂₃ # new_D₂₃ ≤ r₂ + r₃
    end
    return maximum(conditions)
end


function cone_constraint(v)
    return minimum(v)
end


function get_trimetric_contour(i, c, x_range, y_range, G, (X_trans, Y_trans, Z_trans); cut_to_sector=true, args...)
    #=
    V = v₁*X + v₂*Y + v₃*Z
    v = [v₁,v₂,v₃]

    
    
    c = v₁G₁₁ + v₂G₁₂ + v₃G₁₃
    v'Gv = 1
    1 = v₁v₁G₁₁ + v₁v₂G₁₂ + v₁v₃G₁₃
      + v₂v₁G₂₁ + v₂v₂G₂₂ + v₂v₃G₂₃
      + v₃v₁G₃₁ + v₃v₂G₃₂ + v₃v₃G₃₃
      = v₁*(c+v₂G₂₁+v₃G₃₁)
      + v₂v₂G₂₂ + v₂v₃G₂₃
      + v₃v₂G₃₂ + v₃v₃G₃₃
      
      v₁ = (1-v₂v₂G₂₂-v₂v₃G₂₃-v₃v₂G₃₂-v₃v₃G₃₃)/(c+v₂G₂₁+v₃G₃₁)

    Gramian is normalized:
    G₁₁=G₂₂=G₃₃=1.0
    Gramian is symmetric:
    G₁₂=G₂₁, G₂₃=G₃₂, G₁₃=G₃₁

    v₁ = (1- v₂² - 2v₂v₃G₂₃ -v₃²)/(c + v₂G₂₁ + v₃G₃₁)
    
    c = (v₂G₁₂ + v₃G₁₃) + (1-v₂²-2v₂v₃G₂₃-v₃²)/(c+v₂G₁₂+v₃G₁₃)
    0 = v₂²(G₁₂²-1)+2v₂v₃(G₁₂G₁₃-G₂₃)+v₃²(G₁₃²-1) + (1-c²)
    
    Solve implicit function for v₂ & v₃, compute v₁
    
    Compute great circle distances:
    [d₁ d₂ d₃]' = arccos.(G[v₁ v₂ v₃]')

    Use distances to triangulate position of contour    
    =#

    v_shifted = if c^2 ≈ 1.0
        [[1.0-eps(Float64)],[0.0],[0.0]]
    else
        G_shifted = circshift(G,(1-i,1-i))
        G₁₂,G₁₃,G₂₃ = G_shifted[1,2],G_shifted[1,3],G_shifted[2,3]

        f(v₂,v₃) = v₂^2 * (G₁₂^2 - 1.0) + 2.0 * v₂*v₃*(G₁₂*G₁₃-G₂₃) + v₃^2 * (G₁₃^2 - 1.0) + 1.0-c^2

        compute_v₁(v₂,v₃) = (1.0 - v₂^2 - 2v₂*v₃*G₂₃ -v₃^2)/(c+v₂*G₁₂+v₃*G₁₃)

        function constraint(v₂,v₃)
            v_shifted = [compute_v₁(v₂,v₃),v₂,v₃]
            v_unshifted=circshift(v_shifted,i-1)
            return if !all(-1.0 .≤ (G*v_unshifted) .≤ 1.0)
                -1
            elseif cut_to_sector
                cone_constraint(v_unshifted)
            else
                trimetric_constraint(v_unshifted,X_trans,Y_trans,Z_trans,G)
            end
        end
        v₂,v₃ = implicit_function_contour(f,x_range,y_range; constraint, args...)

        valid = constraint.(v₂,v₃) .≥ 0
        v₂,v₃=v₂[valid],v₃[valid]
        [compute_v₁.(v₂,v₃),v₂,v₃]
    end

    v₁,v₂,v₃ = circshift(v_shifted,i-1)

    if isempty(v₁)
        return (x=Float64[],y=Float64[])
    end

    D = collect(eachcol(acos.(G*[v₁';v₂';v₃'])))
    res = triangulate.(D,Ref(X_trans),Ref(Y_trans),Ref(Z_trans))
    
    if !cut_to_sector
        distance_to_edge(x,p1,p2) = (x[1] - p2[1]) * (p1[2] - p2[2]) - (p1[1] - p2[1]) * (x[2] - p2[2])

        delete_res = zeros(Bool,length(res))
        last_was_nan = false
        for (i,x) in enumerate(res)
            a=distance_to_edge(x,X_trans,Y_trans)
            b=distance_to_edge(x,Y_trans,Z_trans)
            c=distance_to_edge(x,Z_trans,X_trans)

            if !all(>(0), (a,b,c)) && !all(≤(0), (a,b,c))
                if last_was_nan
                    delete_res[i]=true
                else
                    res[i] = (NaN,NaN)
                    last_was_nan=true
                end
            else
                last_was_nan = false
            end
        end
        deleteat!(res, delete_res)
    end

    if isempty(res)
        return (x=Float64[],y=Float64[])
    end

    (x, y) = eachcol([collect.(res)'...;])


    if !isempty(x) && ( isnan(first(x)) || isnan(first(y)) )
        x=x[2:end]
        y=y[2:end]
    end

    if !isempty(x) && ( isnan(last(x)) || isnan(last(y)) )
        x=x[1:end-1]
        y=y[1:end-1]
    end

    return (;x,y)
end

