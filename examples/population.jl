using DPC, StatsBase, Distributions, SparseArrays, DataFrames

input_dim = 100
num_words = 10
spikes_per_word = 30
target_sequence_length = 10
num_sequences = 10

subsequence_length = 3
num_subsequences_per_target = 10
num_synapses_per_word = 20
num_neurons_per_subsequence = spikes_per_word

metasequence_length = 3
num_metasequences_per_target = 10
num_synapses_per_subsequence = num_synapses_per_word

neurons_input = input_dim
neurons_first_layer = num_sequences*num_subsequences_per_target*num_neurons_per_subsequence
neurons_second_layer = num_sequences*num_metasequences_per_target
num_neurons = neurons_input + neurons_first_layer + neurons_second_layer

P_syn = 0.5
θ_syn = 15

jitter_dist = Gamma(1.0,1.0)
syllable_dist = Gamma(5.0,5.0)
break_dist = Gamma(15.0,15.0)

## generate codebook
codebook = spzeros(Bool, input_dim, num_words+1)
for word in 1:num_words
    idx = sample(1:input_dim, spikes_per_word; replace=false)
    codebook[idx,word] .= true
end

## generate sequences
sequences = [(sample(1:num_words, target_sequence_length) for i in 1:num_sequences)...;;]
sequence_timings = cumsum(rand(syllable_dist,target_sequence_length,num_sequences), dims=1)

## generate subsequences
subsequence_positions = zeros(Int,num_subsequences_per_target,num_sequences)
subsequences = zeros(Int,subsequence_length,num_subsequences_per_target, num_sequences)
for i in 1:num_sequences
    for n in 1:num_subsequences_per_target
        idx = sample(1:target_sequence_length, subsequence_length; replace=false)
        sort!(idx)
        subsequence_positions[n,i] = idx[end]
        subsequences[:,n,i] = sequences[idx,i]
    end
end

## generate metasequences
metasequences = zeros(Int, metasequence_length, num_metasequences_per_target, num_sequences)
for i in 1:num_sequences
    for n in 1:num_metasequences_per_target
        idx = sample(1:num_subsequences_per_target, metasequence_length; replace=false)
        order=sortperm(subsequence_positions[idx,i])
        metasequences[:,n,i] = idx[order]
    end
end

## generate network
net = Network()

# inputs
for i in 1:input_dim
    Input(Symbol("input_$(i)"), net)
end

# subsequence neurons
subseq_neurons = Dict{Tuple{Int,Int,Int},Neuron}()
for tgt in 1:num_sequences
    for n in 1:num_subsequences_per_target
        subsequence = subsequences[:,n,tgt]
        for i in 1:num_neurons_per_subsequence
            j=1
            root = Neuron(Symbol("subseq_$(tgt)_$(n)_$(i)_1"), net; θ_syn)
            subseq_neurons[(tgt,n,i)]=root

            idx,_ = findnz(codebook[:,subsequence[end+1-j]])
            sub_idx = sample(idx, num_synapses_per_word; replace=false)
            sort!(sub_idx)
            for (k,s) in enumerate(sub_idx)
                Synapse(Symbol("subseq_$(tgt)_$(n)_$(i)_$(j)_$(k)"), net.inputs[s], root; )
            end

            for j in 2:subsequence_length
                root=Segment(Symbol("subseq_$(tgt)_$(n)_$(i)_$(j)"), root; θ_syn)

                idx,_ = findnz(codebook[:,subsequence[end+1-j]])
                sub_idx = sample(idx, num_synapses_per_word; replace=false)
                sort!(sub_idx)
                for (k,s) in enumerate(sub_idx)
                    Synapse(Symbol("subseq_$(tgt)_$(n)_$(i)_$(j)_$(k)"), net.inputs[s], root)
                end
            end
            root.θ_seg = 0
        end
    end
end

# metasequence neurons
for tgt in 1:num_sequences
    for n in 1:num_metasequences_per_target
        metasequence = metasequences[:, n, tgt]
        j=1
        root = Neuron(Symbol("metaseq_$(tgt)_$(n)_1"), net; θ_syn)

        meta_idx = sample(1:num_neurons_per_subsequence, num_synapses_per_subsequence; replace=false)
        sort!(meta_idx)
        for (k,s) in enumerate(meta_idx)
            Synapse(Symbol("metaseq_$(tgt)_$(n)_$(j)_$(k)"), subseq_neurons[(tgt,metasequence[end+1-j],s)], root)
        end

        for j in 2:subsequence_length
            root=Segment(Symbol("metaseq_$(tgt)_$(n)_$(j)"), root; θ_syn)

            meta_idx = sample(1:num_neurons_per_subsequence, num_synapses_per_subsequence; replace=false)
            sort!(meta_idx)
            for (k,s) in enumerate(meta_idx)
                Synapse(Symbol("metaseq_$(tgt)_$(n)_$(j)_$(k)"), subseq_neurons[(tgt,metasequence[end+1-j],s)], root)
            end
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
        sequence_idx = sample(Weights([2;ones(size(sequences, 2)-1)]))
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

res=draw_spike_sequence(10000.0, codebook, sequences, sequence_timings, noise=10e-3)
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

lookup = Dict(Symbol("subseq_$(tgt)_$(i)_$(j)_1") => ((tgt-1)*num_subsequences_per_target + i-1)*num_neurons_per_subsequence+j for tgt in 1:num_sequences for i in 1:num_subsequences_per_target for j in 1:num_neurons_per_subsequence)
first_order_spike_events[!,:idx] = getindex.(Ref(lookup),first_order_spike_events[!,:object])

lookup = Dict(Symbol("metaseq_$(tgt)_$(i)_1") => (tgt-1)*num_subsequences_per_target+i for tgt in 1:num_sequences for i in 1:num_metasequences_per_target)
target_lookup = Dict(Symbol("metaseq_$(tgt)_$(i)_1") => tgt for tgt in 1:num_sequences for i in 1:num_metasequences_per_target)
second_order_spike_events[!,:idx] = getindex.(Ref(lookup),second_order_spike_events[!,:object])
second_order_spike_events[!,:tgt] = getindex.(Ref(target_lookup),second_order_spike_events[!,:object])

##

fig = Figure(resolution = (0.75textwidth, 0.6textwidth))
ax1 = fig[1,1] = Axis(fig; title="A    Input sequences", titlealign=:left, height=18, backgroundcolor=:transparent)
ax2 = fig[2,1] = Axis(fig; title="B    Input spikes ($(neurons_input) neurons)", titlealign=:left)
ax3 = fig[3,1] = Axis(fig; title="C    Spikes in first layer ($(neurons_first_layer) neurons)", titlealign=:left)
ax4 = fig[4,1] = Axis(fig; title="D    Spikes in second layer ($(neurons_second_layer) neurons)", titlealign=:left, xlabel="time [ms]", xticks=LinearTicks(11))
hidedecorations!(ax1)

scatter!(ax1, Point2f0.(res.sequence_times, 0.0), markersize=16, color=res.sequence)
annotations!(ax1, string.(res.sequence), Point2f0.(res.sequence_times, 0.0), align=(:center,:center),color=:white, offset=(-1,1), textsize=14)

scatter!(ax2, res.times[res.sequence_indices .== 0], res.input_indices[res.sequence_indices .== 0], markersize=1, color=:gray)
scatter!(ax2, res.times[res.sequence_indices .!= 0], res.input_indices[res.sequence_indices .!= 0], markersize=1, color=res.sequence_indices[res.sequence_indices .!= 0])

scatter!(ax3, first_order_spike_events[!,:t], first_order_spike_events[!,:idx], color=:gray, markersize=1)

sequence_time_ranges = extrema(sequence_timings, dims=1)
for (t,s) in zip(res.sequence_times, res.sequence)
    poly!(ax4,Rect(t+sequence_time_ranges[s][1],(s-1)*num_metasequences_per_target, sequence_time_ranges[s][2]-sequence_time_ranges[s][1], num_metasequences_per_target), color=:silver)
end
scatter!(ax4, second_order_spike_events[!,:t], second_order_spike_events[!,:idx], color=second_order_spike_events[!,:tgt], markersize=2)

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

fig

## save result
save(joinpath(@__DIR__, "figures","population_simulation.png"), fig)
save(joinpath(@__DIR__, "figures","population_simulation.svg"), fig)
save(joinpath(@__DIR__, "figures","population_simulation.pdf"), fig)

##

fig2 = Figure(resolution = (textwidth, 0.6textwidth))


ax_t = fig2[1,1:11] = Axis(fig2; title="A    Target sequence", titlealign=:left, height=30, backgroundcolor=:transparent)
hidedecorations!(ax_t)

s = String["1"]
p = Point2f0[(0,0)]
for (i,c) in enumerate(sequences[:,1])
    push!(s, i==1 ? ": " : "→", string(Char('A' + c-1)))
    push!(p, (i-0.5,0), (i,0))
end

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
