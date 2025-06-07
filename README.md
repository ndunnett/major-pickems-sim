# Major Pick'ems Simulator

This is a basic Python script to simulate tournament stage outcomes for Counter-Strike major tournaments, used to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes.

Each team's [regional standings](https://github.com/ValveSoftware/counter-strike_regional_standings) global ranking points are used to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

### Command line interface

```
usage: simulate.py [-h] -f F [-n N] [-k K] [-s S]

options:
  -h, --help  show this help message and exit
  -f F        path to input data (.json)
  -n N        number of iterations to run
  -k K        number of cores to use
  -s S        sigma value to use for win probability
```

### JSON data format

```
{
  <team name>: {
    "seed": <initial seeding>,
    "rating": <global ranking points>
  }
}
```

### Latest Output: BLAST.tv Austin Major 2025 - Stage 2

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Falcons             63.4%
2.  3DMAX               32.5%
3.  FaZe                27.3%
4.  Virtus.pro          18.0%
5.  HEROIC              15.8%
6.  FURIA               14.0%
7.  MIBR                 9.0%
8.  TYLOO                5.9%
9.  paiN                 5.0%
10. B8                   2.4%
11. Lynn Vision          2.2%
12. OG                   1.1%
13. BetBoom              1.1%
14. Nemiga               0.9%
15. M80                  0.7%
16. Legacy               0.7%

Most likely to 3-1 or 3-2:
1.  HEROIC              64.0%
2.  Virtus.pro          60.3%
3.  FURIA               60.3%
4.  FaZe                60.1%
5.  3DMAX               58.0%
6.  MIBR                53.7%
7.  TYLOO               48.4%
8.  paiN                38.7%
9.  Falcons             35.6%
10. B8                  32.8%
11. Lynn Vision         25.5%
12. OG                  15.3%
13. BetBoom             14.2%
14. Nemiga              12.7%
15. M80                 11.6%
16. Legacy               9.0%

Most likely to 0-3:
1.  Legacy              35.3%
2.  M80                 28.3%
3.  Nemiga              27.4%
4.  BetBoom             27.2%
5.  OG                  24.5%
6.  Lynn Vision         16.8%
7.  B8                  14.1%
8.  paiN                 7.8%
9.  TYLOO                6.0%
10. MIBR                 3.9%
11. FURIA                2.5%
12. HEROIC               2.2%
13. Virtus.pro           2.0%
14. FaZe                 1.2%
15. 3DMAX                0.7%
16. Falcons              0.1%

Run time: 10.52 seconds
```
