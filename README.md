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
1.  PARIVISION            27.0%
2.  Team Liquid           23.1%
3.  100 Thieves           22.4%
4.  HEROIC                20.4%
5.  Ninjas in Pyjamas     18.6%
6.  FaZe Clan             15.8%
7.  Eternal Fire          14.9%
8.  9z Team               10.9%
9.  M80                   10.7%
10. DENDELE                9.0%
11. Imperial               6.8%
12. Wildcard               6.0%
13. FlyQuest               4.8%
14. BC.Game                4.2%
15. Lynn Vision            2.8%
16. Chinggis Warriors      2.6%

Most likely to 3-1 or 3-2:
1.  PARIVISION            52.0%
2.  Team Liquid           50.0%
3.  100 Thieves           49.5%
4.  HEROIC                48.9%
5.  Ninjas in Pyjamas     47.2%
6.  FaZe Clan             44.8%
7.  Eternal Fire          44.5%
8.  M80                   41.5%
9.  DENDELE               40.5%
10. 9z Team               36.8%
11. Imperial              32.9%
12. Wildcard              29.7%
13. FlyQuest              25.3%
14. BC.Game               24.3%
15. Chinggis Warriors     16.6%
16. Lynn Vision           15.4%

Most likely to 0-3:
1.  Chinggis Warriors     30.7%
2.  Lynn Vision           29.5%
3.  BC.Game               22.1%
4.  FlyQuest              20.7%
5.  Wildcard              16.7%
6.  Imperial              15.1%
7.  DENDELE               11.4%
8.  9z Team                9.6%
9.  M80                    9.6%
10. Eternal Fire           6.9%
11. FaZe Clan              6.5%
12. Ninjas in Pyjamas      5.6%
13. HEROIC                 4.7%
14. 100 Thieves            4.1%
15. Team Liquid            3.9%
16. PARIVISION             3.0%

Run time: 0.769 seconds
```

```shell
pickems simulate --file data/2026_singapore_stage_1.toml --report picks
```

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

3-0 picks:
1.  PARIVISION            26.9%
2.  Team Liquid           23.0%

3-1 or 3-2 picks:
1.  100 Thieves           49.4%
2.  HEROIC                48.9%
3.  Ninjas in Pyjamas     47.0%
4.  FaZe Clan             44.9%
5.  Eternal Fire          44.6%
6.  M80                   41.6%

0-3 picks:
1.  Chinggis Warriors     30.6%
2.  Lynn Vision           29.5%

Simulated stars earned: 3.865 +/- 1.423
Expected success (>=5 stars): 32.0%

Run time: 1.483 seconds
```
