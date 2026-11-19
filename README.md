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
pickems simulate --file data/2026_singapore_stage_1.toml
```

### Run the picks report

```shell
pickems simulate --file data/2026_singapore_stage_1.toml --report picks
```

### Assess a set of picks

```shell
pickems simulate --file data/2026_singapore_stage_1.toml --report assess \
  --three-zero "Ninjas in Pyjamas,PARIVISION" \
  --advance "HEROIC,FaZe Clan,Team Liquid,Eternal Fire,100 Thieves,9z Team" \
  --zero-three "Chinggis Warriors,Lynn Vision"

```

### Inspect an input file

```shell
pickems inspect --file data/2026_singapore_stage_1.toml
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

## Latest Output: PGL Singapore 2026 - Stage 1

```shell
pickems simulate --file data/2026_singapore_stage_1.toml
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Ninjas in Pyjamas     21.7%
2.  PARIVISION            21.0%
3.  HEROIC                20.4%
4.  FaZe Clan             19.1%
5.  Team Liquid           18.5%
6.  100 Thieves           17.4%
7.  Eternal Fire          17.2%
8.  9z Team               15.5%
9.  M80                   11.5%
10. DENDELE               10.0%
11. Imperial               7.0%
12. Wildcard               6.5%
13. BC.Game                4.5%
14. FlyQuest               4.1%
15. Chinggis Warriors      2.9%
16. Lynn Vision            2.9%

Most likely to 3-1 or 3-2:
1.  PARIVISION            49.3%
2.  Ninjas in Pyjamas     49.2%
3.  HEROIC                48.1%
4.  FaZe Clan             47.5%
5.  Team Liquid           46.5%
6.  Eternal Fire          45.2%
7.  100 Thieves           45.2%
8.  9z Team               43.4%
9.  M80                   43.1%
10. DENDELE               40.3%
11. Imperial              32.8%
12. Wildcard              30.7%
13. BC.Game               24.1%
14. FlyQuest              21.7%
15. Chinggis Warriors     16.7%
16. Lynn Vision           16.3%

Most likely to 0-3:
1.  Lynn Vision           30.0%
2.  Chinggis Warriors     29.4%
3.  FlyQuest              23.8%
4.  BC.Game               21.5%
5.  Wildcard              16.3%
6.  Imperial              15.0%
7.  DENDELE               10.8%
8.  M80                    9.3%
9.  9z Team                6.9%
10. Eternal Fire           6.1%
11. 100 Thieves            6.0%
12. Team Liquid            5.5%
13. FaZe Clan              5.2%
14. HEROIC                 4.8%
15. PARIVISION             4.7%
16. Ninjas in Pyjamas      4.6%

Run time: 0.794 seconds
```

```shell
pickems simulate --file data/2026_singapore_stage_1.toml --report picks
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  Ninjas in Pyjamas     21.6%
2.  PARIVISION            21.0%

3-1 or 3-2 picks:
1.  HEROIC                48.2%
2.  FaZe Clan             47.6%
3.  Team Liquid           46.4%
4.  Eternal Fire          45.3%
5.  100 Thieves           45.2%
6.  9z Team               43.4%

0-3 picks:
1.  Lynn Vision           29.9%
2.  Chinggis Warriors     29.4%

Simulated stars earned: 3.781 +/- 1.401
Expected success (>=5 stars): 29.1%

Run time: 1.531 seconds
```
