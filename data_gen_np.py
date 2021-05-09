import os
import sys
import json
import time as tl
from math import (exp, log)

import msprime
import numpy as np
# import torch

RHO_LIMIT = (1.6*10e-8, 1.6*10e-10)
MU_LIMIT = (1.25*10e-9, 1.25*10e-7)

LENGTH_NORMALIZE_CONST = 4
ZIPPED = False
NUMBER_OF_EVENTS_LIMITS = (1, 20)
MAX_T_LIMITS = (0.01, 50)
LAMBDA_EXP = 1000
POPULATION_LIMITS = (250, 100000)
POPULATION = 5000

# N = 20
N = int(sys.argv[4])

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


L_HUMAN = 30_000_000
RHO_HUMAN = 1.6*10e-9
MU_HUMAN = 1.25*10e-8

RHO_LIMIT = (1.6*10e-10, 1.6*10e-8)
MU_LIMIT = (1.25*10e-7, 1.25*10e-9)

NUMBER_OF_EVENTS_LIMITS = (1, 20)
LAMBDA_EXP = 20_000

POPULATION = 10_000
POPULATION_COEFF_LIMITS = (0.5, 1.5)
MIN_POPULATION_NUM = 1_000


def give_rho() -> float:
    return np.random.uniform(*RHO_LIMIT)


def give_mu() -> float:
    return np.random.uniform(*MU_LIMIT)


def give_random_coeff(mean=.128, var=.05) -> float:
    return np.random.beta(.1, .028)*.0128


def give_random_rho(base=RHO_HUMAN) -> float:
    return np.random.uniform(0.0001, 100, 1)[0]*base


def generate_demographic_events(population: int = POPULATION) -> 'msprime.Demography':
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=population)

    number_of_events = np.random.randint(*NUMBER_OF_EVENTS_LIMITS)

    times = sorted(np.random.exponential(LAMBDA_EXP, size=number_of_events))

    last_population_size = population
    for t in times:
        last_population_size = max(last_population_size * np.random.uniform(*POPULATION_COEFF_LIMITS),
                                   MIN_POPULATION_NUM)
        demography.add_population_parameters_change(
            t, initial_size=last_population_size)

    return demography


class arg:
    population = 5000.0
    rho = RHO_HUMAN
    mu = MU_HUMAN
    # num_repl = 1
    # l = 300
    num_repl = int(float(sys.argv[2]))
    lengt = int(float(sys.argv[3]))
    ratio_train_examples = 0.9
    random_seed = int(sys.argv[5])
    model = "hudson"
    sample_size = 2
    demographic_events = generate_demographic_events()


def simple_split(time: float, N: int, split_const: int = 5000) -> int:
    return int(min(time//split_const, N-1))


class DataGenerator():
    def __init__(self,
                 recombination_rate: float = RHO_HUMAN,
                 mutation_rate: float = MU_HUMAN,
                 demographic_events: list = None,
                 population: int = None,
                 number_intervals: int = N,
                 splitter=simple_split,  # maust be annotiede
                 num_replicates: int = 1,
                 lengt: int = L_HUMAN,
                 model: str = "hudson",
                 random_seed: int = 42,
                 sample_size: int = 1,
                 ):

        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        if not demographic_events:
            if not population:
                raise BaseException(
                    "Eiter demographic_events or population must be speciefied")
            demographic_events = msprime.Demography()
            demographic_events.add_population(
                name="A", initial_size=population)
        self.demographic_events = demographic_events
        self.splitter = splitter
        self.model = model
        self.len = lengt
        self.random_seed = random_seed
        self.number_intervals = number_intervals
        self._data = None

    def run_simulation(self):
        """
        return generator(tskit.TreeSequence)
        function run the simulation with given parametrs
        """
        self._data = msprime.sim_ancestry(
            recombination_rate=self.recombination_rate,
            sequence_length=self.len,
            num_replicates=self.num_replicates,
            demography=self.demographic_events,
            model=self.model,
            random_seed=self.random_seed,
            samples=self.sample_size)
        return self._data

    def __iter__(self):
        return self

    def __next__(self):
        """
        return haplotype, recombination points and coalescent time
        """
        if self._data is None:
            self.run_simulation()

        try:
            tree = next(self._data)
        except StopIteration:
            raise StopIteration

        mutated_ts = msprime.sim_mutations(
            tree, rate=self.mutation_rate)  # random_seed

        times = [0]*self.len
        mutations = [0]*self.len
        prior_dist = [0.0]*self.number_intervals

        for m in mutated_ts.mutations():
            mutations[int(m.position)] = 1

        for t in mutated_ts.aslist():
            interval = t.get_interval()
            left = interval.left
            right = interval.right
            time = t.get_total_branch_length()/2
            times[int(left):int(right)] = [time]*int(right-left)
            prior_dist[self.splitter(
                time, self.number_intervals)] += (int(right-left))/self.len

        return mutations, times, prior_dist


if __name__ == "__main__":

    num_model = int(sys.argv[6])
    x_path = os.path.join(sys.argv[1], "x")
    y_path = os.path.join(sys.argv[1], "y")
    pd_path = os.path.join(sys.argv[1], "PD")

    name = 0
    print(f'Path: {sys.argv[1]}')
    print(f"Num_model: {num_model}")
    print(f"Num replicates: {sys.argv[2]}")
    for j in range(num_model):
        generator = DataGenerator(
            demographic_events=generate_demographic_events(),
            num_replicates=int(sys.argv[2])
        )
        generator.run_simulation()
        # return mutations, times, None, prior_dist, None
        for x, y, t in generator:
            print(name)
            x = np.array(x, dtype=np.int64)
            y = np.array(y)
            pd = np.array(t)

            np.save(x_path + "/" + str(name), x)
            np.save(y_path + "/" + str(name), y)
            np.save(pd_path + "/" + str(name), pd)
            name += 1
