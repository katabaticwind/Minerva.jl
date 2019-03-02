mutable struct Recorder
    history
    items
    url
end

Recorder(items, url) = Recorder([], items, url)

function stamp!(r::Recorder, message)
    missing_items = setdiff(r.items, collect(keys(message)))
    length(missing_items) > 0 && error("missing items in message: $missing_items")
    push!(r.history, message)
end

function save(r::Recorder)
    # TODO
end

r = Recorder([:episode, :score, :steps, :epsilon], "/Users/colinswaney/Desktop/record.txt")
message = Dict(:episode => 1, :score => 45, :steps => 45, :epsilon => 1.0)
stamp!(r, message)
message = Dict(:episode => 2, :score => 45)
stamp!(r, message)
