[private]
@default:
    just --list

# Run tests with optional filter
@test filter="":
    cargo nextest run --cargo-quiet {{ filter }}

# Run formatter on all files
@fmt:
    cargo fmt --all --quiet

# Run clippy with warnings promoted to errors
@check:
    cargo clippy --quiet --all-targets --all-features -- -D warnings

# Generate profile data for pprof
@generate-profile-data:
    cargo run --profile=pprof --features=pprof

# Run pprof to profile the simulation hot path
@profile:
    just generate-profile-data && pprof -http=localhost:8080 target/profile.pb

# Use criterion to benchmark the simulation hot path
@bench:
    cargo bench --quiet

# Run simulation
@simulate report="basic":
    cargo run --release -- simulate -f data/$(ls data -Art | tail -n 1) -r {{ report }}

# Run simulation in debug mode
@simulate-dev report="basic":
    cargo run -- simulate -f data/$(ls data -Art | tail -n 1) -r {{ report }}

# Run data update command
@data-update:
    cargo run -- data update -p data
