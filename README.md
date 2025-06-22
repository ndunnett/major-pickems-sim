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

```toml
["<string: team name>"]
seed = <integer: initial seed for tournament stage>
rating = <integer: current global ranking points>
...
```

### Latest Output: BLAST.tv Austin Major 2025 - Stage 3

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Vitality              69.7%
2.  MOUZ                  36.5%
3.  Spirit                33.7%
4.  The MongolZ           18.2%
5.  Aurora                11.0%
6.  Natus Vincere          9.3%
7.  G2                     7.2%
8.  Liquid                 4.8%
9.  3DMAX                  3.7%
10. FaZe                   3.1%
11. Virtus.pro             1.3%
12. FURIA                  1.1%
13. paiN                   0.2%
14. Lynn Vision            0.2%
15. Nemiga                 0.0%
16. Legacy                 0.0%

Most likely to 3-1 or 3-2:
1.  Aurora                65.7%
2.  The MongolZ           64.7%
3.  Spirit                59.3%
4.  MOUZ                  57.0%
5.  Natus Vincere         55.9%
6.  G2                    55.3%
7.  FaZe                  50.3%
8.  3DMAX                 47.4%
9.  Liquid                43.7%
10. Vitality              29.5%
11. Virtus.pro            28.3%
12. FURIA                 26.7%
13. paiN                   6.8%
14. Lynn Vision            6.2%
15. Nemiga                 1.8%
16. Legacy                 1.6%

Most likely to 0-3:
1.  Legacy                48.1%
2.  Nemiga                46.4%
3.  paiN                  34.3%
4.  Lynn Vision           29.3%
5.  FURIA                 12.0%
6.  Virtus.pro            11.3%
7.  FaZe                   4.7%
8.  3DMAX                  4.3%
9.  Liquid                 3.3%
10. G2                     2.4%
11. Natus Vincere          1.4%
12. Aurora                 1.2%
13. The MongolZ            0.7%
14. MOUZ                   0.2%
15. Spirit                 0.2%
16. Vitality               0.0%

Run time: 0.88 seconds
```
