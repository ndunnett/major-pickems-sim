# Major Pick'ems Simulator

Simulate tournament stage outcomes for Counter-Strike major tournaments, to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes.

Each team's [regional standings](https://github.com/ValveSoftware/counter-strike_regional_standings) global ranking points are used to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

## Installation

Download the binary from the latest [release](https://github.com/ndunnett/major-pickems-sim/releases), or install from source using cargo with the nightly toolchain:

```shell
cargo +nightly install major-pickems-sim
```

## Common commands

### Open the interactive TUI

```shell
pickems tui
```

### Run a basic simulation

```shell
pickems simulate --file data/2025_budapest_stage_3.toml
```

### Run the picks report

```shell
pickems simulate --file data/2025_budapest_stage_3.toml --report picks
```

### Assess a set of picks

```shell
pickems simulate --file data/2025_budapest_stage_3.toml --report assess \
  --three-zero "FURIA,G2" \
  --advance "MOUZ,Falcons,Vitality,The MongolZ,Team Spirit,Liquid" \
  --zero-three "Imperial,Passion UA"
```

### Inspect an input file

```shell
pickems inspect --file data/2025_budapest_stage_3.toml
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

## Latest Output: IEM Cologne 2026 - Stage 1

```shell
pickems simulate --file data/2026_cologne_stage_1.toml
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  GamerLegion           30.5%
2.  B8                    27.9%
3.  HEROIC                20.7%
4.  BetBoom               19.2%
5.  BIG                   17.4%
6.  M80                   15.0%
7.  MIBR                  13.9%
8.  SINNERS               12.7%
9.  NRG                    8.3%
10. TYLOO                  7.2%
11. Sharks                 6.9%
12. Gaimin Gladiators      6.2%
13. Liquid                 4.5%
14. Lynn Vision            4.0%
15. THUNDERdOWNUNDER       3.3%
16. FlyQuest               2.2%

Most likely to 3-1 or 3-2:
1.  B8                    51.5%
2.  GamerLegion           51.3%
3.  HEROIC                49.6%
4.  BetBoom               49.1%
5.  BIG                   47.0%
6.  M80                   44.5%
7.  MIBR                  43.2%
8.  SINNERS               41.2%
9.  NRG                   38.9%
10. TYLOO                 35.5%
11. Sharks                34.0%
12. Gaimin Gladiators     32.0%
13. Liquid                25.3%
14. Lynn Vision           23.0%
15. THUNDERdOWNUNDER      20.0%
16. FlyQuest              13.9%

Most likely to 0-3:
1.  FlyQuest              32.5%
2.  THUNDERdOWNUNDER      25.6%
3.  Lynn Vision           22.5%
4.  Liquid                20.5%
5.  Gaimin Gladiators     16.0%
6.  Sharks                14.6%
7.  TYLOO                 14.0%
8.  NRG                   12.2%
9.  SINNERS                7.8%
10. MIBR                   7.3%
11. M80                    6.7%
12. BIG                    5.6%
13. BetBoom                4.9%
14. HEROIC                 4.4%
15. B8                     2.8%
16. GamerLegion            2.5%

Run time: 0.026 seconds
```

```shell
pickems simulate --file data/2026_cologne_stage_1.toml --report picks
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  GamerLegion           30.4%
2.  MIBR                  13.9%

3-1 or 3-2 picks:
1.  B8                    51.5%
2.  HEROIC                49.5%
3.  BetBoom               49.1%
4.  BIG                   46.9%
5.  M80                   44.7%
6.  SINNERS               41.2%

0-3 picks:
1.  FlyQuest              32.6%
2.  THUNDERdOWNUNDER      25.6%

Simulated stars earned: 3.852 +/- 1.397
Expected success (>=5 stars): 31.6%

Run time: 0.05 seconds
```
