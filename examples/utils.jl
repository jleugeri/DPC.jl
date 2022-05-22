using DPC
using Colors
import CairoMakie
Makie = CairoMakie.Makie
import ColorTypes: RGB
import DataStructures: DefaultDict, OrderedDict
CairoMakie.activate!()

export statetrace!, statetrace, StateTrace

const textwidth = 469.75499 / 0.75

function make_manual_ticks(manual_ticks, manual_labels)
    @assert length(manual_ticks) == length(manual_labels)
    idx = sortperm(manual_ticks)
    t = copy(manual_ticks)
    l = copy(manual_labels)
    t[idx], x->l[idx]
end


pal = [colorant"#fcaf3e",colorant"#729fcf",colorant"#ad7fa8",colorant"#ef5e29",colorant"#98e289",colorant"#fce94f"]


mytheme = Makie.Theme(
    fontsize = 12,
    font = "Linux Libertine O",
    palette = (color=pal,),
    Axis = (
        backgroundcolor = :gray90,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        xgridcolor = :white,
        ygridcolor = :white,
        xminorgridcolor = :white,
        yminorgridcolor = :white,
        titlesize = 14,
        xticklabelsize = 10,
        yticklabelsize = 10,
        xlabelsize = 12,
        ylabelsize = 12,
        xlabelpadding = 5f0,
        ylabelpadding = 5f0,
        xticklabelcolor = :gray10,
        yticklabelcolor = :gray10,
        xtickcolor = :gray10,
        ytickcolor = :gray10,
    )
)

Makie.set_theme!(mytheme)

presentation_theme = copy(mytheme)
presentation_theme[:Axis][:titlesize][] = 28
presentation_theme[:Axis][:xticklabelsize][] = 24
presentation_theme[:Axis][:yticklabelsize][] = 24
presentation_theme[:Axis][:xlabelsize][] = 26
presentation_theme[:Axis][:ylabelsize][] = 26
presentation_theme[:fontsize][] = 26


color_1 = pal[1]
color_1_50 = Makie.RGBAf0(color_1.r,color_1.g,color_1.b,0.5)
color_1_25 = Makie.RGBAf0(color_1.r,color_1.g,color_1.b,0.25)
color_2 = pal[2]
color_2_50 = Makie.RGBAf0(color_2.r,color_2.g,color_2.b,0.5)
color_2_25 = Makie.RGBAf0(color_2.r,color_2.g,color_2.b,0.25)
color_3 = pal[3]
color_3_50 = Makie.RGBAf0(color_3.r,color_3.g,color_3.b,0.5)
color_3_25 = Makie.RGBAf0(color_3.r,color_3.g,color_3.b,0.25)
color_4 = pal[4]
color_4_50 = Makie.RGBAf0(color_4.r,color_4.g,color_4.b,0.5)
color_4_25 = Makie.RGBAf0(color_4.r,color_4.g,color_4.b,0.25)

"""
Computes diameters for branches to satisfy Rall's condition.
"""
function get_branch_diameters(tree::NeuronOrSegment{ID,T,WT,IT}; root_diam=1.0, diams=Dict{ID,Float64}()) where {ID,T,WT,IT}
    diams[tree.id] = root_diam
    child_diam = (root_diam^(3/2)/length(tree.next_upstream))^(2/3)
    for child in tree.next_upstream
        get_branch_diameters(child; root_diam=child_diam, diams=diams)
    end
    return diams
end



"""
Computes the depths in segments from the soma for each segment.
"""
function get_root_distances(tree::NeuronOrSegment{ID,T,WT,IT}; start=0, dists = OrderedDict{ID,Int}()) where {ID,T,WT,IT}
    dists[tree.id] = start
    for child in tree.next_upstream
        get_root_distances(child; start=start+1, dists=dists)
    end
    return dists
end


"""P_superthreshold(threshold, event_probabilities, event_weights)

Compute the probability with which a sum of events with given `event_weights` and `event_probabilities` 
would exceed a given `threshold`.
"""
function P_superthreshold(threshold, event_probabilities, event_weights)
    @assert length(event_probabilities) == length(event_weights)
    if isempty(event_probabilities)
        return threshold ≤ 0 ? 1.0 : 0.0
    end
    
    d = zeros(Bool, length(event_weights))
    P = zero(Float64)
    for i in 1:2^length(event_weights)
        digits!(d, i, base=2)
        if event_weights'*d ≥ threshold
            p = prod(ifelse.(d, event_probabilities, one(Float64) .- event_probabilities))
            P += p 
        end
    end
    P
end

"""P_fire(obj, volleys)

Compute the probability with which the `volleys` (provided they occur in the correct temporal sequence) 
would trigger a spike/plateau in `obj`
"""
function P_fire(obj, volleys)
    (probabilities, weights) = get(volleys, obj.id, ([],[]))
    P_syn = P_superthreshold(obj.θ_syn, probabilities, weights)
    return if isempty(obj.next_upstream) 
        P_syn
    else
        P_seg = P_superthreshold(obj.θ_seg, P_fire.(obj.next_upstream, Ref(volleys)), ones(length(obj.next_upstream)))
        P_syn*P_seg
    end
end