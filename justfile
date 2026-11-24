set minimum-version := "1.56.0"
set no-exit-message

[windows]
set shell := ["cmd.exe", "/D", "/C"]

[private]
@default:
    just --list

# Run tests with an optional filter
[env("RUSTFLAGS", "-C target_cpu=native")]
[group("development")]
@test *filter:
    cargo nextest run --cargo-quiet -- {{ filter }}
alias t := test

# Run Clippy with warnings promoted to errors
[group("development")]
@check:
    cargo clippy --quiet --all-targets -- -D warnings
alias c := check

# Format all Rust files
[group("development")]
@fmt:
    cargo fmt --all --quiet
alias f := fmt

# Test, check, format
[group("development")]
@tcf *filter: (test filter) check fmt

# Generate profile data for pprof
[env("RUSTFLAGS", "-C link-arg=-Wl,--no-rosegment -C force-frame-pointers=yes -C symbol-mangling-version=v0 -C llvm-args=--inline-threshold=0")]
[private]
@_generate-profile-data:
    cargo run --profile=pprof --features=pprof

# Run pprof to profile the simulation hot path
[group("measurement")]
@profile: _generate-profile-data
    pprof -http=localhost:8080 target/profile.pb
alias p := profile

# Run criterion to benchmark the simulation hot path
[group("measurement")]
@bench *args:
    cargo bench --bench criterion --quiet -- {{ args }}
alias b := bench

# Print the size of the final binary
[group("measurement")]
[script("python3")]
size:
    import json
    import sys
    from pathlib import Path

    for line in """{{ quote(`cargo build --quiet --release --message-format=json`) }}""".splitlines():
        if line.startswith('{"reason":"compiler-artifact') and '"crate_types":["bin"],"name":"pickems"' in line:
            path = Path(json.loads(line).get("executable"))
            size = path.stat().st_size / (1000 * 1000)
            print(f"{size:.3f}".rstrip("0").rstrip(".") + " MB")
            sys.exit(0)

    print("failed to read build metadata")
    sys.exit(1)

# Get the path to the most recently modified data file
[private]
[script("python3")]
_latest-data:
    from pathlib import Path
    latest = max((path for path in Path('data').iterdir() if path.is_file()), key=lambda path: path.stat().st_mtime)
    print(latest)

# Run a simulation on the most recently modified data file
[arg("dev", long, short="d", value="", help="runs with the dev build profile")]
[group("quick run")]
@sim dev="--release":
    cargo run {{ dev }} -- simulate --file "{{ `just _latest-data` }}" --report all

# Run the TUI
[arg("dev", long, short="d", value="", help="runs with the dev build profile")]
[group("quick run")]
@tui dev="--release":
    cargo run {{ dev }} -- tui

# Run the data update command
[arg("dev", long, short="d", value="", help="runs with the dev build profile")]
[group("quick run")]
@update dev="--release":
    cargo run {{ dev }} -- update
