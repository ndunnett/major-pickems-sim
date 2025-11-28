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

### Latest Output: StarLadder Budapest Major 2025 - Stage 2

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Aurora                40.8%
2.  Natus Vincere         37.3%
3.  3DMAX                 22.7%
4.  Liquid                22.3%
5.  FaZe                  19.3%
6.  Astralis              18.9%
7.  B8                     8.2%
8.  PARIVISION             5.7%
9.  TYLOO                  5.5%
10. MIBR                   4.3%
11. fnatic                 3.9%
12. Imperial               3.3%
13. Passion UA             2.8%
14. Ninjas in Pyjamas      2.4%
15. FlyQuest               1.6%
16. M80                    0.9%

Most likely to 3-1 or 3-2:
1.  Liquid                60.6%
2.  FaZe                  56.5%
3.  3DMAX                 56.2%
4.  Natus Vincere         54.2%
5.  Astralis              52.5%
6.  Aurora                51.6%
7.  B8                    49.5%
8.  TYLOO                 33.7%
9.  PARIVISION            32.6%
10. fnatic                31.4%
11. MIBR                  25.6%
12. Passion UA            22.9%
13. Imperial              22.0%
14. Ninjas in Pyjamas     21.0%
15. FlyQuest              18.4%
16. M80                   11.3%

Most likely to 0-3:
1.  M80                   36.5%
2.  FlyQuest              27.5%
3.  Ninjas in Pyjamas     22.3%
4.  Passion UA            20.9%
5.  Imperial              17.7%
6.  fnatic                15.9%
7.  MIBR                  14.6%
8.  TYLOO                 11.9%
9.  PARIVISION            11.8%
10. B8                     8.7%
11. Astralis               3.5%
12. FaZe                   2.7%
13. 3DMAX                  2.5%
14. Liquid                 2.1%
15. Natus Vincere          0.8%
16. Aurora                 0.6%

Run time: 0.097 seconds
```
