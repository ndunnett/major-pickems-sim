from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cache, reduce
from itertools import groupby
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from random import random
from time import perf_counter_ns
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator


@dataclass(frozen=True)
class Team:
    id: int
    name: str
    seed: int
    rating: int

    def __str__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return self.id


@dataclass(frozen=True)
class TeamIndex[T]:
    teams: list[Team]
    data: list[T]

    @staticmethod
    def new(teams: Iterable[Team], factory: Callable[[], T]) -> TeamIndex[T]:
        """Create a new index from teams, with default values created by calling `factory`."""
        teams = list(teams)
        data = [factory() for _ in range(len(teams))]
        return TeamIndex(teams, data)

    def __getitem__(self, team: Team) -> T:
        return self.data[team.id]

    def __setitem__(self, team: Team, value: T) -> None:
        self.data[team.id] = value

    def items(self) -> Iterator[tuple[Team, T]]:
        """Returns an iterator of key, value pairs."""
        return ((self.teams[i], self.data[i]) for i in range(len(self.teams)))


@dataclass
class Record:
    wins: int
    losses: int
    teams_faced: set[Team]

    @staticmethod
    def new() -> Record:
        """Create a new empty record."""
        return Record(wins=0, losses=0, teams_faced=set())

    @property
    def diff(self) -> int:
        return self.wins - self.losses


@dataclass
class Result:
    three_zero: int
    advanced: int
    zero_three: int

    @staticmethod
    def new() -> Result:
        """Create a new empty result."""
        return Result(three_zero=0, advanced=0, zero_three=0)

    def __add__(self, other: Result) -> Result:
        return Result(
            three_zero=self.three_zero + other.three_zero,
            advanced=self.advanced + other.advanced,
            zero_three=self.zero_three + other.zero_three,
        )


@cache
def win_probability(a: Team, b: Team, sigma: int) -> float:
    """Calculate the probability of team 'a' beating team 'b' for given sigma value."""
    return 1 / (1 + 10 ** ((b.rating - a.rating) / sigma))


@dataclass
class SwissSystem:
    sigma: int
    records: TeamIndex[Record]
    remaining: set[Team]
    first_round: bool

    @staticmethod
    def new(teams: Iterable[Team], sigma: int) -> SwissSystem:
        """Create a new swiss system from teams and sigma value."""
        return SwissSystem(
            sigma=sigma,
            records=TeamIndex.new(teams, Record.new),
            remaining=set(teams),
            first_round=True,
        )

    def clear(self) -> None:
        self.first_round = True
        self.remaining = set(self.records.teams)

        for record in self.records.data:
            record.teams_faced.clear()
            record.wins = 0
            record.losses = 0

    def seeding(self, team: Team) -> tuple[int, int, int]:
        """Calculate seeding based on win-loss, Buchholz difficulty, and initial seed."""
        return (
            -self.records[team].diff,
            -sum(self.records[opp].diff for opp in self.records[team].teams_faced),
            team.seed,
        )

    def simulate_match(self, team_a: Team, team_b: Team) -> None:
        """Simulate a singular match."""
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
        self.records[team_a].teams_faced.add(team_b)
        self.records[team_b].teams_faced.add(team_a)

        # advance/eliminate teams after best of three
        if is_bo3:
            for team in [team_a, team_b]:
                if self.records[team].wins == 3 or self.records[team].losses == 3:
                    self.remaining.remove(team)

    def simulate_round(self) -> None:
        """Simulate a round of matches."""
        # first round is seeded differently (1-9, 2-10, 3-11 etc.)
        if self.first_round:
            group = sorted(self.remaining, key=lambda team: team.seed)

            for a, b in zip(group, group[len(group) // 2 :]):
                self.simulate_match(a, b)

            self.first_round = False

        # group teams by record and run matches for each group, highest seed vs lowest seed
        else:
            groups = groupby(
                sorted(self.remaining, key=self.seeding),
                lambda team: self.records[team].wins,
            )

            for group in (list(group_iter) for _, group_iter in groups):
                for a, b in zip(group, reversed(group[len(group) // 2 :])):
                    self.simulate_match(a, b)

    def simulate_tournament(self) -> None:
        """Simulate an entire tournament stage."""
        while self.remaining:
            self.simulate_round()


@dataclass(frozen=True)
class Simulation:
    sigma: int
    teams: set[Team]

    @staticmethod
    def from_file(filepath: Path, sigma: int) -> Simulation:
        """Create a new simulation by parsing data from a .json file."""
        with open(filepath) as file:
            data = json.load(file)

        def id_generator() -> Generator[int]:
            i = 0

            while True:
                yield i
                i += 1

        auto_id = id_generator()

        teams = {
            Team(
                id=next(auto_id),
                name=team_k,
                seed=team_v["seed"],
                rating=team_v["rating"],
            )
            for team_k, team_v in data.items()
        }

        return Simulation(sigma, teams)

    def batch(self, n: int) -> TeamIndex[Result]:
        """Run a batch of 'n' simulation iterations for given data and return results."""
        results = TeamIndex.new(self.teams, Result.new)
        ss = SwissSystem.new(self.teams, self.sigma)

        for _ in range(n):
            ss.clear()
            ss.simulate_tournament()

            for team, record in ss.records.items():
                match (record.wins, record.losses):
                    case (3, 0):
                        results[team].three_zero += 1
                    case (3, 1 | 2):
                        results[team].advanced += 1
                    case (0, 3):
                        results[team].zero_three += 1

        return results

    def run(self, n: int, k: int) -> TeamIndex[Result]:
        """Run 'n' simulation iterations across 'k' processes and return results."""
        with Pool(k) as pool:
            m = n // k
            r = n % k
            futures = [pool.apply_async(self.batch, [m + 1 if i < r else m]) for i in range(k)]
            results = [future.get() for future in futures]

        def _f(acc: TeamIndex[Result], results: TeamIndex[Result]) -> TeamIndex[Result]:
            for team, result in results.items():
                acc[team] += result

            return acc

        return reduce(_f, results)


def format_results(results: TeamIndex[Result], iterations: int, run_time: float) -> str:
    """Format simulation results into a readable string."""
    out = [f"RESULTS FROM {iterations:,} TOURNAMENT SIMULATIONS"]

    fields = (
        ("three_zero", "3-0"),
        ("advanced", "3-1 or 3-2"),
        ("zero_three", "0-3"),
    )

    for attr, title in fields:
        out.append(f"\nMost likely to {title}:")

        def result_key(tup: tuple[Team, Result]) -> int:
            return getattr(tup[1], attr)

        sorted_results = enumerate(sorted(results.items(), key=result_key, reverse=True))

        for i, (team, result) in sorted_results:
            out.append(
                f"{str(i + 1) + '.':<4}"
                f"{team.name:<20}"
                f"{round(getattr(result, attr) / iterations * 100, 1):>6}%",
            )

    out.append(f"\nRun time: {run_time:.2f} seconds")
    return "\n".join(out)


if __name__ == "__main__":
    # parse args from CLI
    parser = ArgumentParser()
    parser.add_argument("-f", type=Path, help="path to input data (.json)", required=True)
    parser.add_argument("-n", type=int, default=1_000_000, help="number of iterations to run")
    parser.add_argument("-k", type=int, default=cpu_count(), help="number of cores to use")
    parser.add_argument("-s", type=int, default=800, help="sigma value to use for win probability")
    args = parser.parse_args()

    # run simulations and print formatted results
    start = perf_counter_ns()
    results = Simulation.from_file(args.f, args.s).run(args.n, args.k)
    run_time = (perf_counter_ns() - start) / 1_000_000_000
    print(format_results(results, args.n, run_time))
