using JSON

abstract type AbstractRecorder end

"""
ExperimentRecorder

    A structure to record and write experiment status updates to file.

    Write to JSON file with format:

        "train": {
          [message],
          ...
        },
        "test": {
          [message]
        }

    # Example
    ```julia-repl
    r = ExperimentRecorder([:episode, :score, :steps, :epsilon], "/Users/colinswaney/Desktop/record.txt")
    message = Dict(:episode => 1, :score => 45, :steps => 45, :epsilon => 1.0)
    stamp!(r, message, :train)
    message = Dict(:episode => 2, :score => 24, :steps => 24, :epsilon => 1.0)
    stamp!(r, message, :train)
    message = Dict(:episode => 2, :score => 45)  # this will fail
    stamp!(r, message, :test)
    save(r)
    ```
"""
mutable struct ExperimentRecorder <: AbstractRecorder
    history
    items
    url
end

ExperimentRecorder(items, url) = ExperimentRecorder((train=[], test=[]), items, url)
ExperimentRecorder(url) = ExperimentRecorder([:episode, :score, :steps, :epsilon], url)

function stamp!(r::ExperimentRecorder, message, message_type::Symbol)
    missing_items = setdiff(r.items, collect(keys(message)))
    length(missing_items) > 0 && error("missing items in message: $missing_items")
    push!(r.history[message_type], message)
end

function save(r::ExperimentRecorder)
    open(r.url, "w") do f
        write(f, json(r.history, 2))
    end
end
