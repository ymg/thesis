#!/usr/bin/env python3

from rdata_wrapper_2 import initalize_r_env, get_model_prediction
import copy
import glob
import os
import random
import re
import sys
import warnings
import json
from collections import defaultdict, OrderedDict
from functools import partial
from itertools import chain, islice, tee
from pathlib import Path

import numpy as np
import pandas
from natsort import natsorted

sys.path.append("model_sampling/py_sde_perf")

FULL_DATASET, MC_TREE, CATEGORISED_SET = None, None, None
PROGRAM = ''
INPUT_SIZE = ''

R_MODELS = Path('model_sampling/py_sde_perf')
DIR = Path('c_samples/pthread/csv')

PROGRAMS = (
    ('fasta', 900000), ('fasta', 1000000), ('fasta', 1100000), ('fasta', 1200000),
    ('binarytrees', 20), ('binarytrees',
                          21), ('binarytrees', 22), ('binarytrees', 23),
    ('spectralnorm', 8500), ('spectralnorm',
                             9500), ('spectralnorm', 10500), ('spectralnorm', 11500),
    ('prime_decomposition', 0),
)

FREQUENCIES = [1.2, 1.4, 1.8, 2.1, 2.6]
CORES = list(range(2, 29))


def build_csv():
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for freq in FREQUENCIES:
        os.chdir(DIR / f'{freq}')

        for __file in glob.glob('*.csv'):

            with open(__file) as csvfile:

                file_name = ''
                input_size = -1

                if 'fasta' in __file:
                    file_name = 'fasta'
                    if '1000000' in __file:
                        input_size = 1000000
                    elif '1100000' in __file:
                        input_size = 1100000
                    elif '1200000' in __file:
                        input_size = 1200000
                    elif '900000' in __file:
                        input_size = 900000

                elif 'binarytrees' in __file:
                    file_name = 'binarytrees'
                    if '20' in __file:
                        input_size = 20
                    elif '21' in __file:
                        input_size = 21
                    elif '22' in __file:
                        input_size = 22
                    elif '23' in __file:
                        input_size = 23

                elif 'spectralnorm' in __file:
                    file_name = 'spectralnorm'
                    if '8500' in __file:
                        input_size = 8500
                    elif '9500' in __file:
                        input_size = 9500
                    elif '10500' in __file:
                        input_size = 10500
                    elif '11500' in __file:
                        input_size = 11500

                elif 'prime_decomposition' in __file:
                    file_name = 'prime_decomposition'
                    input_size = 0

                data = pandas.read_csv(csvfile)
                data.rename(columns={'Total Time': 'Total.Time'}, inplace=True)

                d[file_name][freq][input_size] = (
                    file_name, freq, input_size, data)

    return d


def fetch(program, freq, core, input_size):

    if int == type(core):
        # print(f'{program} {freq} {core} {input_size}')
        core = f'N{core}'
        # print(f'{program} {freq} {core} {input_size}')

    if str == type(core) or 'NN' in core:
        rewrite = re.findall(r'\d+', core)
        core = f'N{rewrite[0]}'

    state_tmp = FULL_DATASET[program][freq][input_size]

    try:
        if state_tmp[3].empty:
            pass
    except Exception:
        print(f'{program} {freq} {core} {input_size}')
        print('DataFrame is empty!')
        sys.exit(1)

    state = state_tmp[3].loc[state_tmp[3]['Core'] == str(core)]

    estimated_energy = get_model_prediction('nnls', state)

    return state_tmp[0], state_tmp[1], state_tmp[2], estimated_energy, state


def get_cores(df):
    core = df['Core'].item()
    return core


def get_energy(df):
    if df.empty:
        print(df)
        sys.exit('DataFrame is empty!')

    energy = df['Energy Reading'].item()
    return np.float64(energy)


def prenext(l, v):
    i = l.index(v)
    return l[i - 1] if i > 0 else None, l[i + 1] if i < len(l) - 1 else None


def select_state(current_state):

    current_state_cores = re.findall(r'\d+', get_cores(current_state[4]))

    freq = random.choice([1.2, 1.4, 1.8, 2.1, 2.6])
    core = random.choice(range(int(current_state_cores[0]), 29))

    new_state = fetch(PROGRAM, freq, f'N{core}', INPUT_SIZE)

    if current_state[3] > new_state[3]:
        return new_state
    else:
        return None


def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)


# Hill Climbing simulation on tree
def exec_sim(gen, lowest):
    value = 0
    state = gen

    accepted = None

    i = select_state(state)
    if i:
        value += 1
        state = i
    else:
        value += 1

    if state[3] <= lowest:
        accepted = state

    return accepted


def new_rand_child(state):
    [a, b] = np.random.randint(2, size=2)
    child = {'cores': 0, 'freq': 0}

    if a:
        child['cores'] = get_cores(state[4])
    else:
        child['cores'] = random.choice(range(2, 29))

    if b:
        child['freq'] = state[1]
    else:
        child['freq'] = random.choice([1.2, 1.4, 1.8, 2.1, 2.6])

    new_state = fetch(PROGRAM, child["freq"], f'{child["cores"]}', INPUT_SIZE)

    return new_state


def new_child(parent_one, parent_two):
    [a, b, _] = np.random.randint(2, size=3)
    child = {'cores': 0, 'freq': 0, 'lib': ''}

    swap = np.random.randint(2, size=1)

    if swap:
        parent_one, parent_two = parent_two, parent_one

    if a:
        child['cores'] = get_cores(parent_one[4])
    else:
        child['cores'] = get_cores(parent_two[4])

    if b:
        child['freq'] = parent_one[1]
    else:
        child['freq'] = parent_two[1]

    mutate = np.random.randint(2, size=1)

    if mutate:
        child['freq'] = random.choice(FREQUENCIES)
    else:
        child['cores'] = random.choice(CORES)

    new_state = fetch(PROGRAM, child["freq"], f'{child["cores"]}', INPUT_SIZE)

    return new_state


def pick_n_lowest(states, n=None):
    if n is None:
        n = 0

    return sorted(states, key=lambda x: x[3])[:n]


def get_lowest_energy_across_all_frequencies(dfs):
    global FULL_DATASET, INPUT_SIZE

    lowest_energy_val = None

    for freq_ in FREQUENCIES:
        energy_val = FULL_DATASET[PROGRAM][freq_][INPUT_SIZE][3]['Energy Reading'].min(
        )

        if not lowest_energy_val:
            lowest_energy_val = energy_val

        if lowest_energy_val > energy_val:
            lowest_energy_val = energy_val

    return lowest_energy_val


def run_simulation(file_idx):

    accepted_results, seed_pop = [], []
    oldest_member, best_solution_selection = OrderedDict([]), defaultdict(list)

    lowest_plug_35percent = get_lowest_energy_across_all_frequencies(
        FULL_DATASET[PROGRAM]) * 1.35

    threshold = get_lowest_energy_across_all_frequencies(
        FULL_DATASET[PROGRAM]) * 1.05

    first_solution = None

    number_of_popl_discovered = 0
    for _ in range(10):
        seed_pop.append(fetch(PROGRAM, random.choice(FREQUENCIES),
                        f'N{random.choice(range(1, 29))}', INPUT_SIZE))
        number_of_popl_discovered += len(seed_pop)

    oldest_member['gen0'] = seed_pop

    if not first_solution:
        temp_obj = eligible_solution('gen0', threshold, seed_pop)
        if temp_obj:
            first_solution = temp_obj

    partial_app = partial(exec_sim, lowest=lowest_plug_35percent)
    current_pop = map(partial_app, seed_pop)
    current_pop = [new_child(current, nxt) for _, current, nxt in previous_and_next(
        current_pop) if current and nxt]

    current_pop = rebuild_population(current_pop, current_pop)

    oldest_member['gen1'] = current_pop

    if not first_solution:
        temp_obj = eligible_solution('gen1', threshold, current_pop)
        if temp_obj:
            first_solution = temp_obj

    for num in range(2, 8):
        # p = list(previous_and_next(best_pop))
        partial_app = partial(exec_sim, lowest=lowest_plug_35percent)
        current_pop = map(partial_app, current_pop)
        current_pop = list(filter(None.__ne__, list(current_pop)))
        # current_pop = current_pop if len(current_pop) % 2 == 0 else current_pop[:-1]
        new_pop = [new_child(current, nxt) for _, current, nxt in previous_and_next(
            current_pop) if current and nxt]

        current_pop = rebuild_population(current_pop, new_pop)
        oldest_member[f'gen{num}'] = current_pop

        if not first_solution:
            temp_obj = eligible_solution(f'gen{num}', threshold, current_pop)
            if temp_obj:
                first_solution = temp_obj

    with open(f'sample_{file_idx}.csv', 'a') as f:
        print('Frequency,Cores,Actual_Energy,Estimated_Energy,Exec_Time', file=f)
        for o in current_pop:
            _, frequency, no_cores, estimated_energy, df = o
            print(
                f'{frequency},{no_cores},{df["Energy Reading"].item()},{estimated_energy},{df["Total.Time"].item()}', file=f)

    oldest = defaultdict(lambda: defaultdict(list))
    for gen_no, pop in natsorted(oldest_member.items()):
        flag = 0
        for _gen, _pop in natsorted(oldest_member.items()):
            if gen_no == _gen or flag:
                flag = 1
                t = intersection(pop, _pop)
                if t:
                    oldest[gen_no][_gen] = t
                else:
                    break
            else:
                continue

    oldest_gen = defaultdict(lambda: list)
    last_generation = list(oldest_member)[-1]
    for g, v in oldest.items():
        gen_dict = list(oldest[g][last_generation])
        for _gen_members in gen_dict:
            if last_generation in list(v) and np.float64(oldest[g][last_generation][_gen_members][3]) <= threshold:
                # oldest_gen[g] = oldest[g][last_generation]

                for entry, val in oldest[g][last_generation].items():
                    oldest_gen[g] = [entry, val[0], val[1],
                                     val[2], val[3], get_energy(val[4])]

                break

    if first_solution:
        oldest_gen['first solution'] = (first_solution[0], first_solution[1][0], first_solution[1][1],
                                        first_solution[1][2], first_solution[1][3], get_energy(first_solution[1][4]))

    with open(f'sample_{file_idx}.json', 'a') as f:
        print(json.dumps(oldest_gen, indent=2,
              sort_keys=True, default=myconverter), file=f)


def eligible_solution(gen_num, min_threshold, population):
    lowest_in_population = pick_n_lowest(population, 1)[0]

    if lowest_in_population[3] <= min_threshold:
        return gen_num, lowest_in_population
    else:
        return None


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pandas.datetime.datetime):
        return obj.__str__()


def rebuild_population(current_pop, new_pop):
    remaining = []
    new_pop, remaining = balance_population(current_pop, new_pop, remaining)

    if len(new_pop) + len(remaining) < 10:

        # if remaining population is empty and none were fit to survive,
        try:
            _s = pick_n_lowest(remaining, n=1)[0]
        except Exception:
            # we generate new single member randomly to keep the population at same size
            _s = fetch(PROGRAM, random.choice(FREQUENCIES),
                       f'N{random.choice(range(1, 29))}', INPUT_SIZE)

        # then we go back to random binary selection of specific features (cores, frequency, and library)
        while len(new_pop) < 10:
            new_pop += [new_rand_child(_s)]

        remaining = []
        new_pop, remaining = balance_population(
            current_pop, new_pop, remaining)

    elif len(new_pop) + len(remaining) >= 10:
        remaining = []
        new_pop, remaining = balance_population(
            current_pop, new_pop, remaining)

    adjusted_pop = [*new_pop, *remaining]
    current_pop = copy.copy(adjusted_pop)

    return current_pop


def balance_population(current_pop, new_pop, remaining):
    # new_pop = [i for i in new_pop if not i[3].empty]

    if len(new_pop) <= 9:
        r = 10 - len(new_pop)
        remaining = pick_n_lowest(current_pop, n=r)
    elif len(new_pop) >= 10:
        new_pop = pick_n_lowest(new_pop, n=10)
    return new_pop, remaining


def intersection(lst1, lst2):
    intersected = {}

    for val1, val2, val3, estimated_energy1, df1 in lst1:
        for val4, val5, val6, estimated_energy2, df2 in lst2:
            if val1 == val4 and val2 == val5 and val3 == val6 and get_cores(df1) == get_cores(df2):
                intersected[f'{val1}_{val2}_{val3}_{get_cores(df1)}'] = val1, val2, val3, estimated_energy1, df1

    return intersected


def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)


def main():
    global FULL_DATASET, INPUT_SIZE, PROGRAM, CATEGORISED_SET, R_MODELS

    warnings.filterwarnings('ignore')

    FULL_DATASET = build_csv()

    initalize_r_env(f'{R_MODELS}/[PARSEC] Model Data.RData')
    # initalize_r_env(f'{R_MODELS}/Models_Predictions_v4.RData')

    for p, i in PROGRAMS:
        INPUT_SIZE = i
        PROGRAM = p

        os.chdir(DIR / 'genetic-algorithm')

        os.makedirs(f'{p}_{i}', exist_ok=False)

        os.chdir(DIR / 'genetic-algorithm' / f'{p}_{i}')

        print(f'Currently sampling: {p}\tFor Input: {i}')
        print(f'In directory: {os.getcwd()}')

        for i_ in range(1, 1001):
            run_simulation(i_)


if __name__ == "__main__":
    main()
