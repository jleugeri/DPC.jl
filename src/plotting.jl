using Makie
import DataStructures: DefaultDict

export Port

@recipe(Steps, t, y) do scene
    Theme(
        color = :black,
        fill = nothing,
    )
end

function Makie.plot!(myplot::Steps)
    t = myplot[1]
    y = myplot[2]

    line = Observable(Point2f0[])
    function update!(t, y)
        empty!(line[])
        y_old = y[1]
        for (_t,_y) in zip(t,y)
            push!(line[], Point2f0(_t,y_old))
            push!(line[], Point2f0(_t,_y))
            y_old = _y
        end
    end

    Makie.Observables.onany(update!, t, y)
    update!(t[],y[])

    poly!(myplot, line, color=myplot[:fill])
    lines!(myplot, line, color=myplot[:color])
    myplot
end


@recipe(StateTrace, t, state, color, offset, height) do scene
    Theme(
    )
end

function Makie.plot!(plt::StateTrace)
    t,state,color,offset,height=plt[:t],plt[:state],plt[:color],plt.offset,plt.height
    rects = Observable(Rect2D{Float64}[])
    colors = Observable(Any[])
    
    Makie.Observables.onany(t, state, color, offset, height) do t,state,color, offset, height
        t₀,s₀ = first(t),first(state)
        for (t₁,s₁) in zip(t[2:end],state[2:end])
            push!(rects[], Rect(float(t₀), float(offset), float(t₁-t₀), float(height)))
            push!(colors[], color[s₀])
            t₀,s₀ = t₁,s₁
        end
    end

    # Force update
    t[] = t[]

    poly!(plt, rects, color=colors)

    plt
end



struct Port{W,I}
    name::I
end
Port{W}(name::I) where {W,I} = Port{W,I}(name)


function Makie.default_theme(scene::Makie.SceneLike, ::Type{<: Makie.Plot(Neuron)})
    Theme(
        color=RGBAf0(0.9,0.9,0.9,1.0),
        branch_width=0.1,
        branch_length=1.0,
        angle_between=10/180*pi,
        xticks = [],
        root_position = Point2f0(0,0),
        ports = Dict{Symbol,Vector{Symbol}}(),
        fxaa = false,
    )
end

function Makie.plot!(treeplot::Makie.Plot(Neuron))
    # get the actual object to plot
    tree = to_value(treeplot[1])
    # get position of the root = offset of all nodes
    offset = treeplot[:root_position]
    
    # if obj is an observable dict-type holding named values, get named value ...
    maybe_get(obj::Observable, key) = if isa(obj[],Union{DefaultDict,Dict})
        @lift $obj[key]
    else
        obj
    end

    # if obj is a dict-type holding named values, get named value ...
    maybe_get(obj::Union{DefaultDict,Dict}, key) = obj[key]
    # ... otherwise return the obj itself
    maybe_get(obj, key) = obj

    
    function serialize_tree(tree::NeuronOrSegment{ID,T,WT,IT}, l, w, ω, given_ports) where {ID,T,WT,IT}
        sectors = Float64[]
        depths = Float64[]
        all_points = Vector{Pair{ID,Point2f0}}[]
        all_ports = Vector{Pair{ID,Point2f0}}[]
        all_points_flat = Pair{ID,Point2f0}[]
        all_ports_flat = Pair{ID,Point2f0}[]
        all_parents_flat = Pair{ID,ID}[]

        # go through all branches once to collect information
        for subtree in tree.next_upstream
            # get angle and depth of sector
            α,d,points,ports,parents = serialize_tree(subtree, l, w, ω, given_ports)
            # compute depth of new sector
            ll = maybe_get(l, subtree.id)
            c = √(ll^2+d^2-2*ll*d*cos(π-α/2))
            # compute angle of new sector
            β=2acos((ll^2+c^2-d^2)/(2ll*c))

            push!(sectors, β+ maybe_get(ω, subtree.id))
            push!(depths, c)
            push!(all_points, points)
            push!(all_ports, ports)
            append!(all_parents_flat, parents)
            push!(all_parents_flat, subtree.id=>tree.id)
        end

        # calculate total angle and depth for current (sub-)tree
        sector = sum(sectors)
        depth = isempty(depths) ? 0.0 : maximum(depths)

        # calculate angles for all branches
        branch_angles = cumsum(sectors) .- sectors./2 .- sector/2

        for (branch, branch_angle, points, ports) in zip(tree.next_upstream, branch_angles, all_points, all_ports)
            # shift by ll and rotate by `branch_angle`
            ll = maybe_get(l, branch.id)
            c=cos(branch_angle)
            s=sin(branch_angle)
            transform = ((x,y),) -> Point2f0(c*x-s*(y+ll),s*x+c*(y+ll))
    
            branch_end = transform(Point2f0(0.0,0.0))
            # add branch for branch
            push!(all_points_flat, branch.id=>branch_end)
    
            # add ports for branch
            branch_ports = get(given_ports, branch.id, ID[])
            append!(all_ports_flat, Pair.(branch_ports, LinRange(Point2f0(0,0), branch_end, length(branch_ports)+2)[2:end-1]))
    
            # keep branches from branches' subtree
            if !isempty(points)            
                append!(all_points_flat, map(((name,pts),) -> name=>transform(pts), points))
            end
    
            # keep ports from branches' subtree
            if !isempty(ports)            
                append!(all_ports_flat, map(((prt,pos),) -> prt=>transform(pos), ports))
            end
        end

        if isa(tree, Neuron)
            neuron_ports = get(given_ports, tree.id, ID[])

            ww = maybe_get(w, tree.id)
            append!(all_ports_flat, Pair.(neuron_ports, LinRange(Point2f0(0,-ww), Point2f0(0,ww), length(neuron_ports)+2)[2:end-1]))
        end

        return (sector=sector, depth=depth, nodes=all_points_flat, ports=all_ports_flat, parents=all_parents_flat)
    end



    # get all nodes, their ports and parents in flat format
    serialized = lift(treeplot[:branch_length],treeplot[:branch_width],treeplot[:angle_between], treeplot[:ports]) do l,w,ω,prts
        ser=serialize_tree(tree,l,w,ω,prts)
        push!(ser.nodes, tree.id => Point2f0(0.0,0.0))
        ser
    end
    parent = DefaultDict(tree.id, serialized[].parents...)
    # get all the positions of the ports
    ports = lift(serialized,offset) do ser,off
        [k=>v+off for (k,v) in ser.ports]
    end

    # plot branches
    for (name,parent_name) in serialized[].parents
        c  = maybe_get(treeplot[:color], name)
        w1 = maybe_get(treeplot[:branch_width], name)
        w2 = maybe_get(treeplot[:branch_width], parent_name)

        # dynamically recompute polygon
        branch_poly = lift(w1, w2, serialized, offset) do w1,w2,ser,off
            node_dict = Dict(ser.nodes...)
            b1 = node_dict[name]
            b2 = node_dict[parent_name]
            normal = [0 1; -1 0]*(b2-b1)
            normal /= sqrt(normal'*normal)*2
            Ref(off) .+ Point2f0[
                -normal*w1+b1,
                 normal*w1+b1,
                 normal*w2+b2,
                -normal*w2+b2
            ]
        end
        
        # draw polygon for the branch
        poly!(treeplot, branch_poly, color=c, strokewidth = 1, strokecolor=c)

        # draw circles to cap off branch
        poly!(treeplot, lift(w1,serialized, offset) do w1,ser,off
            node_dict = Dict(ser.nodes...)
            b2 = node_dict[name]
            decompose(Point2f0, Circle(off + b2, w1/2))
        end, color=c, strokewidth = 1, strokecolor=c)

        poly!(treeplot, lift(w1,offset) do w1,off
            decompose(Point2f0, Circle(off, w1/2))
        end, color=c, strokewidth = 1, strokecolor=c)
    end

    # draw polygon for the root
    c = maybe_get(treeplot[:color], tree.id)
    w = maybe_get(treeplot[:branch_width], tree.id)
    root_poly = lift(w,offset) do w,offset
        Point2f0[(-2w/2*√(3), -2w/2), (0,2w), (2w/2*√(3), -2w/2)].+offset
    end
    poly!(treeplot, root_poly, color=c, strokewidth = 1, strokecolor=c)

    treeplot.attributes[:ports] = ports
    treeplot
end

