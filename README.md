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

### Latest Output: BLAST.tv Austin Major 2025 - Stage 3

```text
RESULTS FROM 1,000,000 TOURNAMENT SIMULATIONS

Most likely to 3-0:
1.  Vitality              69.6%
2.  MOUZ                  36.6%
3.  Spirit                33.7%
4.  The MongolZ           18.2%
5.  Aurora                11.0%
6.  Natus Vincere          9.2%
7.  G2                     7.2%
8.  Liquid                 4.7%
9.  3DMAX                  3.7%
10. FaZe                   3.1%
11. Virtus.pro             1.3%
12. FURIA                  1.1%
13. paiN                   0.2%
14. Lynn Vision            0.2%
15. Nemiga                 0.0%
16. Legacy                 0.0%

Most likely to 3-1 or 3-2:
1.  The MongolZ           67.0%
2.  Aurora                65.9%
3.  Spirit                60.6%
4.  Natus Vincere         59.2%
5.  MOUZ                  58.1%
6.  G2                    56.7%
7.  Liquid                48.0%
8.  FaZe                  45.5%
9.  3DMAX                 45.2%
10. Vitality              29.8%
11. Virtus.pro            25.9%
12. FURIA                 23.4%
13. Lynn Vision            5.9%
14. paiN                   5.8%
15. Nemiga                 1.5%
16. Legacy                 1.5%

Most likely to 0-3:
1.  Legacy                48.2%
2.  Nemiga                46.4%
3.  paiN                  34.2%
4.  Lynn Vision           29.3%
5.  FURIA                 12.1%
6.  Virtus.pro            11.3%
7.  FaZe                   4.7%
8.  3DMAX                  4.3%
9.  Liquid                 3.3%
10. G2                     2.4%
11. Natus Vincere          1.5%
12. Aurora                 1.2%
13. The MongolZ            0.7%
14. MOUZ                   0.2%
15. Spirit                 0.2%
16. Vitality               0.0%

Run time: 10.52 seconds
```
