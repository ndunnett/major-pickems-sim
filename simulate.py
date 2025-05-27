from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cache, reduce
from multiprocessing import Pool
from os import cpu_count
from random import random
from statistics import median
from time import perf_counter_ns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Team:
    name: str
    seed: int
    rating: tuple[int, ...]

    def __repr__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Team) -> bool:  # type: ignore
        return self.name == other.name


@dataclass
class Record:
    wins: int
    losses: int

    @property
    def diff(self) -> int:
        return self.wins - self.losses


@cache
def win_probability(a: Team, b: Team, sigma: tuple[int, ...]) -> float:
    """Calculate the probability of team 'a' beating team 'b' for given sigma values."""
    # calculate the win probability for given team ratings and value of sigma (std deviation of
    # ratings) for each rating system (assumed to be elo based and normally distributed) and
    # take the median
    return median(
        1 / (1 + 10 ** ((b.rating[i] - a.rating[i]) / (2 * sigma[i]))) for i in range(len(sigma))
    )


@dataclass
class SwissSystem:
    sigma: tuple[int, ...]
    records: dict[Team, Record]
    faced: dict[Team, set[Team]]
    remaining: set[Team]
    finished: set[Team]

    def seeding(self, team: Team) -> tuple[int, int, int]:
        """Calculate seeding based on win-loss, Buchholz difficulty, and initial seed."""
        return (
            -self.records[team].diff,
            -sum(self.records[opp].diff for opp in self.faced[team]),
            team.seed,
        )

    def simulate_match(self, team_a: Team, team_b: Team) -> None:
        """Simulate singular match."""
        # BO3 if match is for advancement/elimination
        is_bo3 = self.records[team_a].wins == 2 or self.records[team_a].losses == 2

        # calculate single map win probability
        p = win_probability(team_a, team_b, self.sigma)

        # simulate match outcome
        if is_bo3:
            first_map = p > random()
            second_map = p > random()
            team_a_win = p > random() if first_map != second_map else first_map
        else:
            team_a_win = p > random()

        # update team records
        if team_a_win:
            self.records[team_a].wins += 1
            self.records[team_b].losses += 1
        else:
            self.records[team_a].losses += 1
            self.records[team_b].wins += 1

        # add to faced teams
        self.faced[team_a].add(team_b)
        self.faced[team_b].add(team_a)

        # advance/eliminate teams after best of three
        if is_bo3:
            for team in [team_a, team_b]:
                if self.records[team].wins == 3 or self.records[team].losses == 3:
                    self.remaining.remove(team)
                    self.finished.add(team)

    def simulate_round(self) -> None:
        """Simulate round of matches."""
        even_teams, pos_teams, neg_teams = [], [], []

        # group teams with the same record together and sort by mid-round seeding
        for team in sorted(self.remaining, key=self.seeding):
            if self.records[team].diff > 0:
                pos_teams.append(team)
            elif self.records[team].diff < 0:
                neg_teams.append(team)
            else:
                even_teams.append(team)

        # first round is seeded differently (1-9, 2-10, 3-11 etc.)
        if len(even_teams) == len(self.records):
            half = len(even_teams) // 2
            even_teams[half:] = reversed(even_teams[half:])

        # run matches for each group, highest seed vs lowest seed
        for group in [pos_teams, even_teams, neg_teams]:
            half = len(group) // 2

            for a, b in zip(group[:half], reversed(group[half:])):
                self.simulate_match(a, b)

    def simulate_tournament(self) -> None:
        """Simulate entire tournament stage."""
        while self.remaining:
            self.simulate_round()


class Simulation:
    sigma: tuple[int, ...]
    teams: set[Team]

    def __init__(self, filepath: Path) -> None:
        """Parse data loaded in from .json file."""
        with open(filepath) as file:
            data = json.load(file)

        self.sigma = (*data["sigma"].values(),)
        self.teams = {
            Team(
                team_k,
                team_v["seed"],
                tuple((eval(sys_v))(team_v[sys_k]) for sys_k, sys_v in data["systems"].items()),  # noqa: S307
            )
            for team_k, team_v in data["teams"].items()
        }

    def batch(self, n: int) -> dict[Team, dict[str, int]]:
        """Run batch of 'n' simulation iterations for given data and return results."""
        results = {team: {stat: 0 for stat in ["3-0", "3-1 or 3-2", "0-3"]} for team in self.teams}

        for _ in range(n):
            ss = SwissSystem(
                self.sigma,
                {team: Record(0, 0) for team in self.teams},
                {team: set() for team in self.teams},
                set(self.teams),
                set(),
            )

            ss.simulate_tournament()

            for team, record in ss.records.items():
                if record.wins == 3:
                    if record.losses == 0:
                        results[team]["3-0"] += 1
                    else:
                        results[team]["3-1 or 3-2"] += 1
                elif record.wins == 0:
                    results[team]["0-3"] += 1

        return results

    def run(self, n: int, k: int) -> dict[Team, dict[str, int]]:
        """Run 'n' simulation iterations across 'k' processes and return results."""
        with Pool(k) as pool:
            futures = [pool.apply_async(self.batch, [n // k]) for _ in range(k)]
            results = [future.get() for future in futures]

        def _f(acc: dict, res: dict) -> dict:
            for team, result in res.items():
                for key, val in result.items():
                    acc[team][key] += val
            return acc

        return reduce(_f, results)


def format_results(results: dict[Team, dict[str, int]], n: int, run_time: float) -> list[str]:
    """Formats simulation results and run time parameters into readable string."""
    out = [f"RESULTS FROM {n:,} TOURNAMENT SIMULATIONS"]

    for stat in next(iter(results.values())):
        out.append(f"\nMost likely to {stat}:")

        for i, (team, result) in enumerate(
            sorted(results.items(), key=lambda tup: tup[1][stat], reverse=True),
        ):
            out.append(
                f"{str(i + 1) + '.':<3} {team.name:<18} {round(result[stat] / n * 100, 1):>5}%",
            )

    out.append(f"\nRun time: {run_time:.2f} seconds")
    return out


if __name__ == "__main__":
    # parse args from CLI
    parser = ArgumentParser()
    parser.add_argument("-f", type=str, help="path to input data (.json)", required=True)
    parser.add_argument("-n", type=int, default=1_000_000, help="number of iterations to run")
    parser.add_argument("-k", type=int, default=cpu_count(), help="number of cores to use")
    args = parser.parse_args()

    # run simulations and print formatted results
    start = perf_counter_ns()
    results = Simulation(args.f).run(args.n, args.k)
    run_time = (perf_counter_ns() - start) / 1_000_000_000
    print("\n".join(format_results(results, args.n, run_time)))
