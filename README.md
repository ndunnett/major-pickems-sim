# Major Pick'ems Simulator

This is a basic program to simulate tournament stage outcomes for Counter-Strike major tournaments, used to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes.

Each team's [regional standings](https://github.com/ValveSoftware/counter-strike_regional_standings) global ranking points are used to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

### Installation

Download the binary from the latest [release](https://github.com/ndunnett/major-pickems-sim/releases), or install from source using cargo:

```shell
cargo install major-pickems-sim
```

### Command line interface

#### Simulate tournament outcomes

```text
Usage: pickems simulate [OPTIONS] --file <FILE>

Options:
  -f, --file <FILE>              Path to load input data from (.toml)
  -n, --iterations <ITERATIONS>  Number of iterations to run [default: 1000000]
  -s, --sigma <SIGMA>            Sigma value to use for win probability [default: 800]
  -h, --help                     Print help
```

#### Use the data wizard to create an input data file

```text
Usage: pickems data wizard --file <FILE>

Options:
  -f, --file <FILE>  Path to save input data to (.toml)
  -h, --help         Print help
```

#### Inspect input data file

```text
Usage: pickems data inspect --file <FILE>

Options:
  -f, --file <FILE>  Path to load input data from (.toml)
  -h, --help         Print help
```

### TOML input data format

The input data file uses the TOML format, with a section for each team containing the initial seed and global ranking points for that team. Each data file is expected to contain exactly 16 teams.

```toml
["{string: team name}"]
seed = {integer: initial seed for tournament stage}
rating = {integer: current global ranking points}
```

### Latest Output: StarLadder Budapest Major 2025 - Stage 1

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Legacy                49.9%
2.  FaZe                  37.9%
3.  B8                    25.7%
4.  GamerLegion           23.0%
5.  fnatic                17.5%
6.  PARIVISION            14.2%
7.  Ninjas in Pyjamas      8.0%
8.  Imperial               7.8%
9.  FlyQuest               3.8%
10. Lynn Vision            3.0%
11. M80                    2.6%
12. RED Canids             1.6%
13. The Huns               1.5%
14. NRG                    1.4%
15. Fluxo                  1.2%
16. Rare Atom              0.8%

Most likely to 3-1 or 3-2:
1.  GamerLegion           61.3%
2.  B8                    59.6%
3.  FaZe                  54.2%
4.  PARIVISION            53.3%
5.  fnatic                52.3%
6.  Legacy                46.3%
7.  Imperial              45.3%
8.  Ninjas in Pyjamas     44.6%
9.  FlyQuest              39.9%
10. Lynn Vision           32.5%
11. M80                   30.1%
12. RED Canids            18.4%
13. Fluxo                 18.0%
14. The Huns              17.4%
15. NRG                   16.3%
16. Rare Atom             10.5%

Most likely to 0-3:
1.  Rare Atom             35.1%
2.  Fluxo                 28.4%
3.  NRG                   25.5%
4.  The Huns              24.7%
5.  RED Canids            22.2%
6.  M80                   16.8%
7.  Lynn Vision           14.8%
8.  FlyQuest              11.3%
9.  Ninjas in Pyjamas      5.8%
10. Imperial               5.6%
11. PARIVISION             3.2%
12. fnatic                 2.9%
13. GamerLegion            1.3%
14. B8                     1.3%
15. FaZe                   0.6%
16. Legacy                 0.3%

Run time: 0.096 seconds
```
