# Major Pick'ems Simulator

This is a basic Python script to simulate tournament stage outcomes for Counter-Strike major tournaments, used to assist decision making for pick'ems. The swiss system follows the seeding rules and format [documented by Valve](https://github.com/ValveSoftware/counter-strike/blob/main/major-supplemental-rulebook.md#seeding), and the tournament rounds are progressed with randomised match outcomes. Each team's ranking from various sources is aggregated to approximate a win probability for each head to head match up. This is by no means an exhaustive or accurate analysis but may give insight to some teams which have higher probability of facing weaker teams to get their 3 wins, or vice versa.

### Command line interface

```
usage: python simulate.py [-h] -f F [-n N] [-k K]

options:
  -h, --help  show this help message and exit
  -f F        path to input data (.json)
  -n N        number of iterations to run
  -k K        number of cores to use
```

### JSON data format

```
{
    "systems": {
        <system name>: <transfer function>
    },
    "sigma": {
        <system name>: <standard deviation for rating>
    },
    "teams": {
        <team name>: {
            "seed": <initial seeding>,
            <system name>: <system rating>
        }
    }
}
```

### Latest Output: BLAST.tv Austin Major 2025 - Stage 2

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Falcons             47.2%
2.  3DMAX               27.7%
3.  FaZe                23.4%
4.  Virtus.pro          18.5%
5.  HEROIC              16.4%
6.  FURIA               15.4%
7.  MIBR                12.0%
8.  TYLOO                8.9%
9.  paiN                 8.1%
10. B8                   4.9%
11. Lynn Vision          4.5%
12. OG                   3.1%
13. BetBoom              2.9%
14. Nemiga               2.7%
15. M80                  2.3%
16. Legacy               2.1%

Most likely to 3-1 or 3-2:
1.  FaZe                55.1%
2.  HEROIC              54.7%
3.  3DMAX               54.5%
4.  Virtus.pro          52.4%
5.  FURIA               52.0%
6.  Falcons             47.7%
7.  MIBR                47.6%
8.  TYLOO               43.2%
9.  paiN                38.0%
10. B8                  32.5%
11. Lynn Vision         28.4%
12. OG                  21.2%
13. BetBoom             20.2%
14. Nemiga              19.1%
15. M80                 18.2%
16. Legacy              15.3%

Most likely to 0-3:
1.  Legacy              28.7%
2.  M80                 24.3%
3.  BetBoom             23.7%
4.  Nemiga              23.4%
5.  OG                  22.0%
6.  Lynn Vision         17.2%
7.  B8                  15.7%
8.  paiN                10.0%
9.  TYLOO                8.8%
10. MIBR                 6.4%
11. FURIA                4.9%
12. HEROIC               4.6%
13. Virtus.pro           4.2%
14. FaZe                 3.1%
15. 3DMAX                2.3%
16. Falcons              0.6%

Run time: 13.43 seconds
```
