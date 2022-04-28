using DPC, StatsBase, Distributions, SparseArrays, DataFrames

input_dim = 100
num_words = 10
spikes_per_word = 30
target_sequence_length = 10
num_distractor_sequences = 9

subsequence_length = 3
num_subsequences = 10
num_synapses_per_word = 20
num_neurons_per_subsequence = spikes_per_word

metasequence_length = 3
num_metasequences = 10
num_synapses_per_subsequence = num_synapses_per_word

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
sequences = [(sample(1:num_words, target_sequence_length) for i in 1:num_distractor_sequences+1)...;;]
sequence_timings = cumsum(rand(syllable_dist,target_sequence_length,num_distractor_sequences+1), dims=1)

## generate subsequences
subsequence_positions = zeros(Int,num_subsequences)
subsequences = zeros(Int,subsequence_length,num_subsequences)
for n in 1:num_subsequences
    idx = sample(1:target_sequence_length, subsequence_length; replace=false)
    sort!(idx)
    subsequence_positions[n] = idx[end]
    subsequences[:,n] = sequences[idx,1]
end

## generate metasequences
metasequences = zeros(Int, metasequence_length, num_metasequences)
for n in 1:num_metasequences
    idx = sample(1:num_subsequences, metasequence_length; replace=false)
    order=sortperm(subsequence_positions[idx])
    metasequences[:,n] = idx[order]
end

## generate network
net = Network()

# inputs
for i in 1:input_dim
    Input(Symbol("input_$(i)"), net)
end

# subsequence neurons
subseq_neurons = Dict{Tuple{Int,Int},Neuron}()
for n in 1:num_subsequences
    subsequence = subsequences[:,n]
    for i in 1:num_neurons_per_subsequence
        j=1
        root = Neuron(Symbol("subseq_$(n)_$(i)_1"), net; θ_syn)
        subseq_neurons[(n,i)]=root

        idx,_ = findnz(codebook[:,subsequence[end+1-j]])
        sub_idx = sample(idx, num_synapses_per_word; replace=false)
        sort!(sub_idx)
        for (k,s) in enumerate(sub_idx)
            Synapse(Symbol("subseq_$(n)_$(i)_$(j)_$(k)"), net.inputs[s], root; )
        end

        for j in 2:subsequence_length
            root=Segment(Symbol("subseq_$(n)_$(i)_$(j)"), root; θ_syn)

            idx,_ = findnz(codebook[:,subsequence[end+1-j]])
            sub_idx = sample(idx, num_synapses_per_word; replace=false)
            sort!(sub_idx)
            for (k,s) in enumerate(sub_idx)
                Synapse(Symbol("subseq_$(n)_$(i)_$(j)_$(k)"), net.inputs[s], root)
            end
        end
        root.θ_seg = 0
    end
end

# metasequence neurons
for n in 1:num_metasequences
    metasequence = metasequences[:, n]
    j=1
    root = Neuron(Symbol("metaseq_$(n)_1"), net; θ_syn)

    meta_idx = sample(1:num_neurons_per_subsequence, num_synapses_per_subsequence; replace=false)
    sort!(meta_idx)
    for (k,s) in enumerate(meta_idx)
        Synapse(Symbol("metaseq_$(n)_$(j)_$(k)"), subseq_neurons[(metasequence[end+1-j],s)], root)
    end

    for j in 2:subsequence_length
        root=Segment(Symbol("metaseq_$(n)_$(j)"), root; θ_syn)

        meta_idx = sample(1:num_neurons_per_subsequence, num_synapses_per_subsequence; replace=false)
        sort!(meta_idx)
        for (k,s) in enumerate(meta_idx)
            Synapse(Symbol("metaseq_$(n)_$(j)_$(k)"), subseq_neurons[(metasequence[end+1-j],s)], root)
        end
    end 

    root.θ_seg = 0
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
        sequence_idx = rand(1:size(sequences, 2))
        push!(sequence,sequence_idx)
        push!(sequence_times,time)

        inp,word_idx,_ = findnz(codebook[:,sequences[:, sequence_idx]])
        t = time .+ sequence_timings[word_idx,sequence_idx] .+ rand(jitter_dist,length(word_idx))
        time = maximum(t) + rand(break_dist)
        if time > duration
            filter!(≤(duration), t)
            time=duration
        end

        
        order = sortperm(t)
        append!(all_spike_times, t[order])
        append!(all_spike_inputs, inp[order])
        append!(all_spike_sequences, fill(sequence_idx, length(t)))
    end
    return (times=all_spike_times, input_indices = all_spike_inputs, sequence_indices = all_spike_sequences, sequence=sequence, sequence_times=sequence_times)
end

res=draw_spike_sequence(10000.0, codebook, sequences, sequence_timings)
#append!(all_spikes, Event.(:input_spikes, 0.0, t, inp))

## run
input=[Event(:input_spikes, 0.0, t, net.inputs[inp]) for (t, inp) in zip(res.times, res.input_indices)]

logger=simulate!(net, input)

spike_events = filter(:event => ==(:spikes),logger.data)
gd=groupby(spike_events, :object; sort=true)

fig = Figure()
ax1 = fig[1,1] = Axis(fig)
ax2 = fig[2,1] = Axis(fig)
scatter!(ax1, res.times, res.input_indices, markersize=2, color=res.sequence_indices)
scatter!(ax2, spike_events[gd.idx,:t], gd.groups, markersize=2, color=gd.groups)
annotations!(ax1, string.(res.sequence), Point2f.(res.sequence_times, Ref(120)))
ylims!(ax1,0,150)
linkxaxes!(ax1,ax2)
fig
