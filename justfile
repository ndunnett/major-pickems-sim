set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

[private]
@default:
    just --list

# Run tests with optional filter
[env("RUSTFLAGS", "-C target_cpu=native")]
@test *filter:
    cargo nextest run --cargo-quiet {{ filter }}

# Run formatter on all files
@fmt:
    cargo fmt --all --quiet

# Run clippy with warnings promoted to errors
@check:
    cargo clippy --quiet --all-targets -- -D warnings

# Generate profile data for pprof
[env("RUSTFLAGS", "-C link-arg=-Wl,--no-rosegment -C force-frame-pointers=yes -C symbol-mangling-version=v0 -C llvm-args=--inline-threshold=0")]
@generate-profile-data:
    cargo run --profile=pprof --features=pprof

# Run pprof to profile the simulation hot path
@profile: generate-profile-data
    pprof -http=localhost:8080 target/profile.pb

# Use criterion to benchmark the simulation hot path
@bench *args:
    cargo bench --quiet -- {{ args }}

[private]
[script("python3")]
_latest-data:
    from pathlib import Path
    latest = max((path for path in Path('data').iterdir() if path.is_file()), key=lambda path: path.stat().st_mtime)
    print(latest)

# Run simulation
@simulate report="basic":
    cargo run --release -- simulate --file "{{ `just _latest-data` }}" --report "{{ report }}"

# Run simulation in debug mode
@simulate-dev report="basic":
    cargo run -- simulate --file "{{ `just _latest-data` }}" --report "{{ report }}"

# Run data update command
@data-update:
    cargo run -- update

# Test, check, format
@tcf *filter:
    just test {{ filter }}
    just check
    just fmt

[script("python3")]
_size:
    import sys
    from pathlib import Path
    path = Path('target/release/pickems.exe' if sys.platform == 'win32' else 'target/release/pickems')
    size = path.stat().st_size / (1024 * 1024)
    print(f'{size:.2f}'.rstrip('0').rstrip('.') + 'M')

# Print the size of the final binary in megabytes
@size:
    cargo build --release
    just _size
