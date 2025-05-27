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

### Sample output

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  HEROIC              50.8%
2.  Complexity          22.5%
3.  FlyQuest            16.9%
4.  Wildcard            16.8%
5.  Nemiga              12.3%
6.  NRG                 11.8%
7.  B8                  10.6%
8.  Imperial            10.3%
9.  BetBoom             10.3%
10. Fluxo                8.1%
11. Legacy               7.6%
12. Metizport            7.4%
13. Lynn Vision          4.8%
14. OG                   4.4%
15. TYLOO                4.0%
16. Chinggis Warriors    1.4%

Most likely to 3-1 or 3-2:
1.  Complexity          51.2%
2.  FlyQuest            51.0%
3.  Wildcard            49.9%
4.  Nemiga              45.5%
5.  HEROIC              44.2%
6.  NRG                 44.0%
7.  Imperial            40.9%
8.  B8                  40.6%
9.  BetBoom             40.1%
10. Fluxo               39.7%
11. Metizport           37.6%
12. Legacy              35.3%
13. Lynn Vision         24.6%
14. OG                  23.5%
15. TYLOO               21.5%
16. Chinggis Warriors   10.5%

Most likely to 0-3:
1.  Chinggis Warriors   37.8%
2.  TYLOO               22.4%
3.  OG                  21.2%
4.  Lynn Vision         19.4%
5.  Metizport           12.1%
6.  Legacy              11.6%
7.  Fluxo               11.4%
8.  BetBoom             11.1%
9.  B8                  10.7%
10. Imperial            10.1%
11. Nemiga               8.0%
12. NRG                  7.8%
13. Wildcard             5.9%
14. FlyQuest             5.6%
15. Complexity           4.4%
16. HEROIC               0.6%

Run time: 13.26 seconds
```
