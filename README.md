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

### Latest Output: StarLadder Budapest Major 2025 - Stage 3

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  FURIA                 41.5%
2.  Falcons               33.3%
3.  Vitality              31.6%
4.  MOUZ                  20.2%
5.  The MongolZ           19.6%
6.  G2                    11.8%
7.  Team Spirit           10.0%
8.  Liquid                 7.3%
9.  Natus Vincere          6.3%
10. paiN                   4.5%
11. 3DMAX                  4.2%
12. B8                     3.4%
13. FaZe                   3.0%
14. PARIVISION             1.9%
15. Passion UA             0.9%
16. Imperial               0.6%

Most likely to 3-1 or 3-2:
1.  MOUZ                  54.5%
2.  Falcons               53.9%
3.  Vitality              53.2%
4.  The MongolZ           51.7%
5.  FURIA                 51.1%
6.  Team Spirit           49.8%
7.  G2                    47.0%
8.  Liquid                44.5%
9.  Natus Vincere         44.5%
10. paiN                  31.6%
11. 3DMAX                 29.5%
12. B8                    29.0%
13. FaZe                  25.4%
14. PARIVISION            18.7%
15. Passion UA             9.2%
16. Imperial               6.5%

Most likely to 0-3:
1.  Imperial              40.9%
2.  Passion UA            36.5%
3.  PARIVISION            24.0%
4.  FaZe                  18.5%
5.  B8                    16.4%
6.  3DMAX                 15.4%
7.  paiN                  11.7%
8.  Natus Vincere          9.4%
9.  Liquid                 8.8%
10. Team Spirit            5.3%
11. G2                     4.6%
12. The MongolZ            2.6%
13. MOUZ                   2.5%
14. Vitality               1.5%
15. Falcons                1.2%
16. FURIA                  0.7%

Run time: 0.093 seconds
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  FURIA                 41.6%
2.  G2                    11.8%

3-1 or 3-2 picks:
1.  MOUZ                  54.4%
2.  Falcons               53.9%
3.  Vitality              53.2%
4.  The MongolZ           51.7%
5.  Team Spirit           49.7%
6.  Liquid                44.6%

0-3 picks:
1.  Imperial              40.8%
2.  Passion UA            36.5%

Simulated stars earned: 4.384 +/- 1.398
Expected success (>=5 stars): 46.7%

Run time: 0.197 seconds
```
