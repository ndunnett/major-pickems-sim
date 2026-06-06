# Major Pick'ems Simulator

Simulate tournament stage outcomes for Counter-Strike major tournaments, to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes.

Each team's [regional standings](https://github.com/ValveSoftware/counter-strike_regional_standings) global ranking points are used to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

## Installation

Download the binary from the latest [release](https://github.com/ndunnett/major-pickems-sim/releases), or install from source using cargo:

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
pickems simulate --file data/2026_cologne_stage_2.toml
```

### Run the picks report

```shell
pickems simulate --file data/2026_cologne_stage_2.toml --report picks
```

### Assess a set of picks

```shell
pickems simulate --file data/2026_cologne_stage_2.toml --report assess \
  --three-zero "Spirit,Legacy" \
  --advance "GamerLegion,Astralis,FUT,G2,9z,B8" \
  --zero-three "FlyQuest,M80"
```

### Inspect an input file

```shell
pickems inspect --file data/2026_cologne_stage_2.toml
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

## Latest Output: IEM Cologne 2026 - Stage 2

```shell
pickems simulate --file data/2026_cologne_stage_2.toml
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Spirit                49.8%
2.  Legacy                31.9%
3.  FUT                   20.4%
4.  G2                    17.6%
5.  Astralis              16.4%
6.  9z                    14.8%
7.  GamerLegion           13.3%
8.  paiN                   8.2%
9.  B8                     7.1%
10. Monte                  6.6%
11. BetBoom                4.1%
12. MIBR                   3.7%
13. BIG                    2.4%
14. TYLOO                  2.0%
15. M80                    1.7%
16. FlyQuest               0.2%

Most likely to 3-1 or 3-2:
1.  GamerLegion           56.8%
2.  Astralis              56.3%
3.  Legacy                56.0%
4.  FUT                   51.9%
5.  G2                    51.4%
6.  9z                    49.0%
7.  Spirit                46.1%
8.  B8                    40.7%
9.  paiN                  37.4%
10. BetBoom               35.3%
11. Monte                 33.1%
12. MIBR                  32.4%
13. BIG                   18.9%
14. TYLOO                 16.9%
15. M80                   15.8%
16. FlyQuest               2.1%

Most likely to 0-3:
1.  FlyQuest              59.6%
2.  M80                   25.1%
3.  TYLOO                 24.0%
4.  BIG                   22.1%
5.  MIBR                  14.7%
6.  BetBoom               12.5%
7.  Monte                  9.5%
8.  B8                     8.0%
9.  paiN                   7.5%
10. GamerLegion            3.8%
11. 9z                     3.2%
12. Astralis               3.1%
13. G2                     2.9%
14. FUT                    2.5%
15. Legacy                 1.0%
16. Spirit                 0.3%

Run time: 0.021 seconds
```

```shell
pickems simulate --file data/2026_cologne_stage_2.toml --report picks
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  Spirit                49.8%
2.  Legacy                31.8%

3-1 or 3-2 picks:
1.  GamerLegion           56.7%
2.  Astralis              56.3%
3.  FUT                   51.9%
4.  G2                    51.3%
5.  9z                    49.2%
6.  B8                    40.8%

0-3 picks:
1.  FlyQuest              59.6%
2.  M80                   25.1%

Simulated stars earned: 4.725 +/- 1.499
Expected success (>=5 stars): 55.9%

Run time: 0.041 seconds
```
