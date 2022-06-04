using DPC, StatsBase, Distributions, SparseArrays, DataFrames, CairoMakie

include("utils.jl")

input_dim = 100
num_words = 10
spikes_per_word = 30
target_sequence_length = 10
num_sequences = 10

subsequence_length = 3
num_subsequence_neurons = 3000
num_synapses_per_segment = 20
num_neurons_per_subsequence = spikes_per_word

metasequence_length = 3
num_metasequences_per_target = 10
num_synapses_per_subsequence = num_synapses_per_segment

neurons_input = input_dim
num_metasequence_neurons = num_sequences*num_metasequences_per_target
num_neurons = neurons_input + num_subsequence_neurons + num_metasequence_neurons

P_syn_in = DPC.BernoulliSynapseWeight(1.0,0.5)
P_syn_sub = DPC.BernoulliSynapseWeight(1.0,0.5)
θ_syn = 8
θ_syn_frac = 0.4

total_t = 10_000.0

jitter_dist = Gamma(1.0,1.0)
word_dist = Gamma(10.0,1.0)
break_dist = Gamma(20.0,20.0)

## generate codebook
codebook = spzeros(Bool, input_dim, num_words+1)
for word in 1:num_words
    idx = sample(1:input_dim, spikes_per_word; replace=false)
    codebook[idx,word] .= true
    
end
## generate sequences
sequences = [(sample(1:num_words, target_sequence_length) for i in 1:num_sequences)...;;]
sequence_timings = cumsum(rand(word_dist,target_sequence_length,num_sequences), dims=1)

## generate subsequences
# stores the actual successive words for each subsequence neuron
subsequences = zeros(Int,subsequence_length,num_subsequence_neurons)
for i in 1:num_subsequence_neurons
    seq = rand(1:num_sequences)
    idx = sample(1:target_sequence_length, subsequence_length; replace=false)
    sort!(idx)
    subsequences[:,i] = sequences[idx,seq]
end

## generate metasequences
# stores the *index* of the target sequence for each metasequence neuron
# metasequences preferentially sample later parts of the target sequence,
# because each word in a meta sequence corresponds to the end of a subsequence
metasequences = zeros(Int, metasequence_length, num_metasequences_per_target, num_sequences)
for seq in 1:num_sequences
    for i in 1:num_metasequences_per_target
        idx = sample(1:target_sequence_length, 
            Weights(clamp.((1:target_sequence_length).-subsequence_length.+1,0,Inf)), 
            metasequence_length; replace=false)
        sort!(idx)
        metasequences[:,seq,i] = idx
    end
end

## generate network
net = Network(weight_type=DPC.BernoulliSynapseWeight)

# inputs
for i in 1:input_dim
    Input(Symbol("input_$(i)"), net)
end

# subsequence neurons
subseq_neurons = Neuron[]
for i in 1:num_subsequence_neurons
    root = nothing
    for j in 1:subsequence_length
        if isnothing(root)
            root = Neuron(Symbol("subseq_$(i)_$(j)"), net; θ_syn)
            push!(subseq_neurons,root)
        else
            root = Segment(Symbol("subseq_$(i)_$(j)"), root; θ_syn)
        end

        idx,_ = findnz(codebook[:,subsequences[end+1-j,i]])
        sub_idx = sample(idx, num_synapses_per_segment; replace=false)
        sort!(sub_idx)
        for (k,s) in enumerate(sub_idx)
            Synapse(Symbol("subseq_$(i)_$(j)_$(k)"), net.inputs[s], root; weight=P_syn_in)
        end
    end
    root.θ_seg = 0
end

# metasequence neurons
for tgt in 1:num_sequences
    for n in 1:num_metasequences_per_target
        metasequence = metasequences[:, n, tgt]
        root = nothing
        for j in 1:metasequence_length
            if isnothing(root)
                root = Neuron(Symbol("metaseq_$(tgt)_$(n)_$(j)"), net; θ_syn)
            else
                root = Segment(Symbol("metaseq_$(tgt)_$(n)_$(j)"), root; θ_syn)
            end

            # the valid subsequences must end with the given word
            valid_subseq_range = sequences[1:metasequence[end-j+1],tgt]
            # find all subsequences that lie inside the valid range
            valid_subseqs = findall(eachcol(subsequences)) do seq
                # subsequence must end with the correct word
                if last(seq) != last(valid_subseq_range)
                    return false
                end

                # all subsequence elements must appear in the correct order
                idx_max = length(valid_subseq_range)
                for el in seq[end-1:-1:1]
                    idx_max=findlast(==(el),valid_subseq_range[1:idx_max-1])
                    if isnothing(idx_max)
                        return false
                    end
                end
                return true
            end

            for (k,s) in enumerate(valid_subseqs)
                Synapse(Symbol("metaseq_$(tgt)_$(n)_$(j)_$(k)"), subseq_neurons[s], root; weight = P_syn_sub)
            end
            root.θ_syn = round(Int, length(valid_subseqs)*θ_syn_frac)
        end 

        root.θ_seg = 0
    end
end


##
function draw_spike_sequence(duration, codebook, sequences, sequence_timings; jitter_dist=Gamma(1.0,1.0), noise=0.0)
    all_spike_times = Float64[]
    all_spike_inputs = Int[]
    all_spike_sequences= Int[]
    sequence = Int[]
    sequence_times = Float64[]
    
    time=0
    while time < duration
        sequence_idx = rand(1:num_sequences)
        push!(sequence,sequence_idx)
        
        inp,word_idx,_ = findnz(codebook[:,sequences[:, sequence_idx]])
        t = time .+ sequence_timings[word_idx,sequence_idx] .+ rand(jitter_dist,length(word_idx))
        time = maximum(t) + rand(break_dist)
        if time > duration
            filter!(≤(duration), t)
            time=duration
        end
        
        push!(sequence_times,mean(t))
        
        order = sortperm(t)
        append!(all_spike_times, t[order])
        append!(all_spike_inputs, inp[order])
        append!(all_spike_sequences, fill(sequence_idx, length(t)))
    end

    # draw background spikes
    num_noise_spikes = rand(Poisson(size(codebook,1)*noise*time))
    t_noise_spikes = rand(num_noise_spikes).*duration
    id_noise_spikes = rand(1:size(codebook,1),num_noise_spikes)

    append!(all_spike_times, t_noise_spikes)
    append!(all_spike_inputs, id_noise_spikes)
    append!(all_spike_sequences, zeros(Int, num_noise_spikes))

    order = sortperm(all_spike_times)

    return (times=all_spike_times[order], input_indices = all_spike_inputs[order], sequence_indices = all_spike_sequences[order], sequence=sequence, sequence_times=sequence_times)
end

res=draw_spike_sequence(total_t, codebook, sequences, sequence_timings, noise=10e-3)
#append!(all_spikes, Event.(:input_spikes, 0.0, t, inp))

## run
input=[Event(:input_spikes, 0.0, t, net.inputs[inp]) for (t, inp) in zip(res.times, res.input_indices)]

logger=simulate!(net, input)

# Example frequencies of event types:
#  :upstate_start          => 64127
#  :epsp_starts            => 30111528
#  :epsp_ends              => 30111528
#  :refractory_period_ends => 52929
#  :upstate_ends           => 54398
#  :plateau_starts         => 71634
#  :backprop               => 181234
#  :spikes                 => 52929
#  :plateau_ends           => 71634

##

first_order_spike_events = filter([:event,:object] => (e,o)-> e==:spikes && startswith(string(o),"subseq_"),logger.data)

second_order_spike_events = filter([:event,:object] => (e,o)-> e==:spikes && startswith(string(o),"metaseq_"),logger.data)

lookup = Dict(Symbol("subseq_$(i)_1") => i for i in 1:num_subsequence_neurons)
first_order_spike_events[!,:idx] = getindex.(Ref(lookup),first_order_spike_events[!,:object])

lookup = Dict(Symbol("metaseq_$(tgt)_$(i)_1") => (tgt-1)*num_metasequences_per_target+i for tgt in 1:num_sequences for i in 1:num_metasequences_per_target)
target_lookup = Dict(Symbol("metaseq_$(tgt)_$(i)_1") => tgt for tgt in 1:num_sequences for i in 1:num_metasequences_per_target)
second_order_spike_events[!,:idx] = getindex.(Ref(lookup),second_order_spike_events[!,:object])
second_order_spike_events[!,:tgt] = getindex.(Ref(target_lookup),second_order_spike_events[!,:object])

##

fig = Figure(resolution = (latex_textwidth, 0.6latex_textwidth))
ax1 = fig[1,1] = Axis(fig; title="A    Input sequences", titlealign=:left, height=18, backgroundcolor=:transparent, clip=false)
ax2 = fig[2,1] = Axis(fig; title="B    Input spikes ($(neurons_input) neurons)", titlealign=:left)
ax3 = fig[3,1] = Axis(fig; title="C    Spikes in first layer ($(num_subsequence_neurons) neurons)", titlealign=:left)
ax4 = fig[4,1] = Axis(fig; title="D    Spikes in second layer ($(num_metasequence_neurons) neurons)", titlealign=:left, xlabel="time [ms]", xticks=LinearTicks(11))
ax5 = fig[:,2] = Axis(fig; title="E    Spike pattern for seq. $(res.sequence[end-1])", ylabel="Input spikes", xlabel="time [ms]", spine=:outer,
leftspinevisible = true,
rightspinevisible = true,
bottomspinevisible = true,
topspinevisible = true,)
ax6 = fig[1,3] = Axis(fig; title="F    Code words for seq. $(res.sequence[end-1])", titlealign=:left)
gr = fig[2:end,3] = GridLayout()
ax7 = gr[1,1] = Axis(fig; title="G    Feature combinations", titlealign=:left)
ax8 = gr[2,1] = Axis(fig; title="H    Connectivity", titlealign=:left)

hidedecorations!(ax1)

scatter!(ax1, Point2f0.(res.sequence_times, 0.0), markersize=16, color=dcolors[res.sequence])
annotations!(ax1, string.(res.sequence), Point2f0.(res.sequence_times, 0.0), align=(:center,:center),color=:white, offset=(-1,1), textsize=14)

scatter!(ax2, res.times[res.sequence_indices .== 0], res.input_indices[res.sequence_indices .== 0], markersize=1, color=:gray)
scatter!(ax2, res.times[res.sequence_indices .!= 0], res.input_indices[res.sequence_indices .!= 0], markersize=1, color=dcolors[res.sequence_indices[res.sequence_indices .!= 0]])


scatter!(ax3, first_order_spike_events[!,:t], first_order_spike_events[!,:idx], color=:gray, markersize=1)

sequence_time_ranges = extrema(sequence_timings, dims=1)
for (t,s) in zip(res.sequence_times, res.sequence)
    poly!(ax4,Rect(t+sequence_time_ranges[s][1]-50,(s-1)*num_metasequences_per_target-5, sequence_time_ranges[s][2]-sequence_time_ranges[s][1]+100, num_metasequences_per_target+10), color=:silver)
end
scatter!(ax4, second_order_spike_events[!,:t], second_order_spike_events[!,:idx], color=dcolors[second_order_spike_events[!,:tgt]], markersize=2)


# draw zoom-in of ax2 into ax5
t₁,t₂ = sequence_time_ranges[res.sequence[end-1]]
zoom_tlims = res.sequence_times[end-1] .+ (t₁-t₂-150,t₂-t₁+150)./2
lines!(ax2, Rect(zoom_tlims[1],0,zoom_tlims[2]-zoom_tlims[1],input_dim), fill=nothing, color=:black)

idx1 = [i for (i,x) in enumerate(res.times)
    if zoom_tlims[1] ≤ x ≤ zoom_tlims[2] && res.sequence_indices[i] == 0]
idx2 = [i for (i,x) in enumerate(res.times)
    if zoom_tlims[1] ≤ x ≤ zoom_tlims[2] && res.sequence_indices[i] != 0]

scatter!(ax5, res.times[idx1], res.input_indices[idx1], markersize=5, color=:gray)
scatter!(ax5, res.times[idx2], res.input_indices[idx2], markersize=5, color=dcolors[res.sequence_indices[idx2]])

xlims!(ax5, zoom_tlims)
hideydecorations!(ax5, label=false)

# draw schema
s = String[]
p = Point2f0[]
for (i,c) in enumerate(sequences[:,end-1])
    push!(s, string(Char('A' + c-1)))
    push!(p, (i,0))
end
annotations!(ax6, s, p, align=(:center,:center),color=:black, offset=(0,1), textsize=14)
hidedecorations!(ax6)
hidedecorations!(ax7)
hidedecorations!(ax8)

#ylims!(ax1,0,150)
xlims!(ax1,(-100,10100))
linkxaxes!.(Ref(ax1),(ax2,ax3,ax4))
hideydecorations!(ax2)
hideydecorations!(ax3)
hideydecorations!(ax4)
hidexdecorations!(ax2, grid=false)
hidexdecorations!(ax3, grid=false)

rowsize!(fig.layout, 2, Relative(0.25))
rowsize!(fig.layout, 3, Relative(0.4))
rowsize!(fig.layout, 4, Relative(0.25))
colsize!(fig.layout, 2, Relative(0.25))
colsize!(fig.layout, 3, Relative(0.25))
rowsize!(gr, 2, Relative(0.3))

fig

## save result
save(joinpath(@__DIR__, "figures","population_simulation.png"), fig)
save(joinpath(@__DIR__, "figures","population_simulation.svg"), fig)
save(joinpath(@__DIR__, "figures","population_simulation.pdf"), fig)

##

fig2 = Figure(resolution = (latex_textwidth, 0.6latex_textwidth))


ax_t = fig2[1,1:11] = Axis(fig2; title="A    Target sequence", titlealign=:left, height=30, backgroundcolor=:transparent)
hidedecorations!(ax_t)


scatter!(ax_t, p[3:2:end], color=sequences[:,1], markersize=16)
scatter!(ax_t, [p[1]], color=[color_1], markersize=16)
annotations!(ax_t, s, p, align=(:center,:center),color=repeat([:white,:black],outer=target_sequence_length+1)[1:end-1], offset=(0,1), textsize=14)
xlims!(ax_t,(-1,11))


ax1 = fig2[2,1] = Axis(fig2; title="B    Code words", titlealign=:left, height=18, backgroundcolor=:transparent)
hidedecorations!(ax1)
ax2 = fig2[2,2] = Axis(fig2; title="C    Sequences of code words", titlealign=:left, height=18, backgroundcolor=:transparent)
hidedecorations!(ax2)


scatter!(ax1, Point2f0.(1:num_words, 0.0), markersize=16, color=1:10)
annotations!(ax1, [string(Char('A'+(i-1))) for i in 1:10], Point2f0.(1:num_words, 0.0), align=(:center,:center),color=:white, offset=(0,1), textsize=14)

ax3 = fig2[3,1] = Axis(fig2; xticks=1:10, ylabel="input spike pattern")
hidexdecorations!(ax3, grid=false)
hideydecorations!(ax3, label=false)

y,x,_=findnz(codebook[:,1:num_words])
scatter!(ax3, x, y, markersize=3, color=x)


for i in 1:(num_sequences)
    ax_i1 = fig2[2,1+i] = Axis(fig2; height=18, backgroundcolor=:transparent)
    hidedecorations!(ax_i1)
    
    t_mean=mean(sequence_timings[:,i])
    scatter!(ax_i1, [Point2f0(t_mean, 0.0)], markersize=16, color= i==1 ? color_1 : :gray)
    annotations!(ax_i1, [string(i)], [Point2f0(t_mean, 0.0)], align=(:center,:center),color=:white, offset=(-1,1), textsize=14)

    ax_i2 = fig2[3,1+i] = Axis(fig2; xticks=([250.0]), xlims=[0,260])
    hideydecorations!(ax_i2)

    y,x,_ = findnz(codebook[:,sequences[:,i]])
    scatter!(ax_i2, sequence_timings[x], y, color=sequences[x,i], markersize=2)

    colsize!(fig2.layout, 1+i, Relative(0.05))

    if i>1
        colgap!(fig2.layout, i, Relative(0.01))
    end
end

ax_x = fig2[3,2:end] = Axis(fig2; xlabel="time [ms]", backgroundcolor=:transparent, xticks=[0.0])
hideydecorations!(ax_x)


gl = fig2[1:3,12] = GridLayout()
ax_s1= gl[1,1] = Axis(fig2; title="D    1st layer", titlealign=:left, xlims=(0,1),ylims=(-num_subsequences_per_target-1,1), backgroundcolor=:transparent)
hidedecorations!(ax_s1)
annotations!(ax_s1, ["S$(Char('₀'+i-1)):  "*join([Char('A' + c-1) for c in col],"→") for (i,col) in enumerate(eachcol(subsequences)) ], Point2f0.(0,-1:-1:-num_subsequences_per_target); textsize=12, align=(:left,:center))
xlims!(ax_s1, (0,1))


ax_s2= gl[2,1] = Axis(fig2; title="E    2nd layer", titlealign=:left, backgroundcolor=:transparent)
hidedecorations!(ax_s2)
annotations!(ax_s2, ["T$(Char('₀'+i-1)):  "*join(["S"*Char('₀' + c-1) for c in col],"→") for (i,col) in enumerate(eachcol(metasequences)) ], Point2f0.(0,-1:-1:-num_metasequences_per_target); textsize=12, align=(:left,:center))
xlims!(ax_s2, (0,1))

colsize!(fig2.layout, 12, Relative(0.2))
rowgap!(fig2.layout, 2, Relative(0.01))

fig2

## save result
save(joinpath(@__DIR__, "figures","population_details.png"), fig2)
save(joinpath(@__DIR__, "figures","population_details.svg"), fig2)
save(joinpath(@__DIR__, "figures","population_details.pdf"), fig2)
