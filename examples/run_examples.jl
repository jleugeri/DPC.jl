using DPC, CairoMakie
include("utils.jl")

files = ["bio_schema.jl", "place_cells.jl", "dithering.jl", "vector_space.jl", "population.jl"]
for file in files
    let
        println("Running $(file) ...")
        include(file)
    end
end

println("DONE.")
