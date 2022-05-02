using DPC, CairoMakie
include("utils.jl")

files = ["bio_schema.jl", "inhibition.jl", "multineuron_and.jl", "multineuron_or.jl", "place_cells.jl", "concept_space.jl", "probabilistic.jl", "motifs.jl"]
for file in files
    let
        println("Running $(file) ...")
        include(file)
    end
end

println("DONE.")
