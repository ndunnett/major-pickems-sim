PROFILING_RUSTFLAGS := "-C link-arg=-Wl,--no-rosegment -C force-frame-pointers=yes -C symbol-mangling-version=v0 -C llvm-args=--inline-threshold=0"
TESTING_RUSTFLAGS := "-C target-cpu=native"

[private]
@default:
    just --list

# Run tests with optional filter
@test filter="":
    RUSTFLAGS='{{ TESTING_RUSTFLAGS }}' cargo nextest run --cargo-quiet {{ filter }}

# Run formatter on all files
@fmt:
    cargo fmt --all --quiet

# Run clippy with warnings promoted to errors
@check:
    cargo clippy --quiet --all-targets -- -D warnings

# Generate profile data for pprof
@generate-profile-data:
    RUSTFLAGS='{{ PROFILING_RUSTFLAGS }}' cargo run --profile=pprof --features=pprof

# Run pprof to profile the simulation hot path
@profile:
    just generate-profile-data && pprof -http=localhost:8080 target/profile.pb

# Use criterion to benchmark the simulation hot path
@bench args="":
    RUSTFLAGS='{{ TESTING_RUSTFLAGS }}' cargo bench --quiet -- {{ args }}

# Run simulation
@simulate report="basic":
    cargo run --release -- simulate --file data/$(ls data -Art | tail -n 1) --report {{ report }}

# Run simulation in debug mode
@simulate-dev report="basic":
    cargo run -- simulate --file data/$(ls data -Art | tail -n 1) --report {{ report }}

# Run data update command
@data-update:
    cargo run -- update

# Test, check, format
@tcf filter="":
    just test {{ filter }} && just check && just fmt

@size build-args="--release":
    cargo build {{ build-args }}; du -A -Bk target/release/pickems
