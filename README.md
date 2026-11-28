# Major Pick'ems Simulator

Simulate tournament stage outcomes for Counter-Strike major tournaments, to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes.

Each team's [regional standings](https://github.com/ValveSoftware/counter-strike_regional_standings) global ranking points are used to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

## Installation

Download the binary from the latest [release](https://github.com/ndunnett/major-pickems-sim/releases), or install the latest release from source using cargo:

```shell
cargo install major-pickems-sim
```

## Common commands

### Open the interactive TUI

```shell
pickems tui
```

### Run a basic simulation

```shell
pickems simulate --file data/2026_singapore_stage_2.toml
```

### Run the picks report

```shell
pickems simulate --file data/2026_singapore_stage_2.toml --report picks
```

### Assess a set of picks

```shell
pickems simulate --file data/2026_singapore_stage_2.toml --report assess \
  --three-zero "The MongolZ,Natus Vincere" \
  --advance "Nemiga,FUT,PARIVISION,G2,Liquid,The One" \
  --zero-three "Lynn Vision,FlyQuest"
```

### Inspect an input file

```shell
pickems inspect --file data/2026_singapore_stage_2.toml
```

### Update local data files from this repository

```shell
pickems update
```

## TOML input data format

Input files contain exactly 16 teams. Each team has an initial seed and rating.

```toml
["{string: team name}"]
seed = {integer: initial seed for tournament stage}
rating = {integer: current global ranking points}
```

## Latest Output: PGL Singapore 2026 - Stage 2

```shell
pickems simulate --file data/2026_singapore_stage_2.toml
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  The MongolZ           34.8%
2.  Natus Vincere         29.7%
3.  FUT                   20.6%
4.  Nemiga                18.2%
5.  G2                    13.8%
6.  The One               10.4%
7.  Liquid                10.3%
8.  PARIVISION            10.2%
9.  100 Thieves            9.4%
10. Astralis               9.2%
11. Ninjas in Pyjamas      8.4%
12. MIBR                   7.8%
13. Eternal Fire           7.3%
14. Imperial               4.4%
15. FlyQuest               3.6%
16. Lynn Vision            1.9%

Most likely to 3-1 or 3-2:
1.  Natus Vincere         54.7%
2.  The MongolZ           50.5%
3.  Nemiga                48.3%
4.  FUT                   47.4%
5.  PARIVISION            45.6%
6.  G2                    44.7%
7.  Liquid                41.0%
8.  The One               37.8%
9.  Ninjas in Pyjamas     37.2%
10. 100 Thieves           36.4%
11. Astralis              34.4%
12. MIBR                  33.7%
13. Eternal Fire          32.1%
14. Imperial              21.9%
15. FlyQuest              20.6%
16. Lynn Vision           13.8%

Most likely to 0-3:
1.  Lynn Vision           35.2%
2.  FlyQuest              24.0%
3.  Imperial              21.8%
4.  Eternal Fire          14.7%
5.  MIBR                  13.5%
6.  Ninjas in Pyjamas     13.0%
7.  100 Thieves           12.7%
8.  Astralis              11.5%
9.  Liquid                10.6%
10. PARIVISION            10.3%
11. The One               10.3%
12. G2                     7.8%
13. Nemiga                 5.6%
14. FUT                    4.8%
15. Natus Vincere          2.4%
16. The MongolZ            1.9%

Run time: 0.015 seconds
```

```shell
pickems simulate --file data/2026_singapore_stage_2.toml --report picks
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  The MongolZ           34.8%
2.  Natus Vincere         29.7%

3-1 or 3-2 picks:
1.  Nemiga                48.4%
2.  FUT                   47.4%
3.  PARIVISION            45.5%
4.  G2                    44.7%
5.  Liquid                40.9%
6.  The One               37.9%

0-3 picks:
1.  Lynn Vision           35.2%
2.  FlyQuest              24.1%

Simulated stars earned: 3.884 +/- 1.457
Expected success (>=5 stars): 33.4%

Run time: 0.018 seconds
```

## Development

Clone the repository and use the [devcontainer](.devcontainer/), it has everything ready to go. Otherwise, ensure the following is installed:

- [Rust](https://rustup.rs/) >=1.95
- [OpenSSL development headers](https://docs.rs/openssl/0.10.82/openssl/) and [pkg-config](https://wiki.freedesktop.org/Software/pkg-config/) (Linux only)
- [just](https://just.systems/man/en/installation.html) >=1.56
- [Python](https://www.python.org/downloads/) >=3.6
- [cargo-nextest](https://nexte.st/) >=0.9.58
- [Samply](https://github.com/mstange/samply) (for profiling)

### Tasks

Project tasks are set up as [just recipes](https://just.systems/man/en/introduction.html), run `just` at the repository root to list them all.

#### Development recipes

- `just test <filter>`: runs cargo-nextest with the given filter, runs all tests if no filter is given
- `just check`: runs Clippy with warnings promoted to errors
- `just fmt`: runs rustfmt on all files
- `just tcf <filter>`: runs `just test <filter>`, `just check`, and `just fmt`

#### Measurement recipes

- `just bench <args>`: runs criterion with the given arguments (see [criterion-rs command-line options](https://criterion-rs.github.io/book/user_guide/command_line_options.html))
- `just profile`: runs the profiler on a simulation benchmark with Samply and serves the profile viewer
- `just size`: builds the project in release mode and prints the size of the final binary

#### Quick run recipes (all run in release mode unless given `--dev` as a flag)

- `just data-update`: runs the update command
- `just sim`: runs a simulation on the most recently modified data file
- `just tui`: starts the TUI
