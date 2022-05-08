#!/usr/bin/env python3

from rdata_wrapper_2 import initalize_r_env, get_model_prediction
import copy
import csv
import glob
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from random import choice

import pandas
from numba import jit

sys.path.append("model_sampling/py_sde_perf")

lowercase = string.ascii_lowercase

FULL_DATASET = None

# upper_energy_scale = 25_000

# DS_ROOT = Path('/Users/ymg/iCloud Drive (Archive) - 1/work/Corryvreckan/data-plots')

R_MODELS = Path('model_sampling/py_sde_perf')
DIR = Path('model_sampling/c_samples/tbb/csv')

PROGRAMS = (
    ('fasta', 900000), ('fasta', 1000000), ('fasta', 1100000), ('fasta', 1200000),
    ('binarytrees', 20), ('binarytrees',
                          21), ('binarytrees', 22), ('binarytrees', 23),
    ('spectralnorm', 8500), ('spectralnorm',
                             9500), ('spectralnorm', 10500), ('spectralnorm', 11500),
    ('prime_decomposition', 0),
)

FREQUENCIES = [1.2, 1.4, 1.8, 2.1, 2.6]


def prenext(l, v):
    i = l.index(v)
    return l[i - 1] if i > 0 else None, l[i + 1] if i < len(l) - 1 else None


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


def generate_neighbour_states(freq, core, best, prog, i_size):

    # random search function body

    # new_state = []
    #
    # f_ = choice(freq)
    # c_ = choice(core)
    #
    # freqs = prenext(freq, f_) if prenext(freq, f_) else None
    # cores = prenext(core, c_) if prenext(core, c_) else None
    #
    # freqs = [x for x in freqs if x]
    # cores = [x for x in cores if x]
    #
    # try:
    #     new_state += [fetch(prog, choice(freqs), f'N{choice(cores)}', i_size)]
    # except ValueError:
    #     pass
    #
    # return new_state

    # hill climbing gen function body

    new_state = []

    x = choice([freq, core])
    cycler = None
    current_set = None

    if freq == x:
        cycler = freq
        current_set = freq

    if core == x:
        cycler = core
        current_set = core

    p = choice(cycler)
    v = prenext(current_set, p) if prenext(current_set, p) else None

    v = list(filter(None, v))

    if any(x in freq for x in v):
        try:
            new_state += [fetch(prog, choice(v), get_cores(best[4]), i_size)]
        except ValueError:
            pass

    if any(x in core for x in v):
        try:
            new_state += [fetch(prog, best[1], choice(v), i_size)]
        except ValueError:
            pass

    return new_state


def search(initial_state, file_idx, input_s, prog, steps=None):

    best_state = initial_state

    if not steps:
        steps = 84

    f = open(f'sample_{file_idx}.csv', 'w')

    write = csv.writer(f)

    write.writerow(['Frequency', 'No. of Cores',
                   'Actual Energy', 'Estimated Energy'])

    for i in range(steps):

        freq = [1.2, 1.4, 1.8, 2.1, 2.6]
        core = list(range(2, 29))
        prob_size = input_s

        prev_new_state = generate_neighbour_states(
            freq, core, best_state, prog, prob_size)

        tmp = []

        for v1 in prev_new_state:

            if v1:
                energy_val = v1[3]
                current = best_state[3]
                if current > energy_val:
                    tmp += [v1]

        if tmp:
            l_ = [e for e in tmp if e]
            p = copy.copy(choice(l_))
            best_state = p

        write.writerow([best_state[1], get_cores(best_state[4]),
                       get_energy(best_state[4]), best_state[3]])

    f.close()

    return best_state


def fetch(program, freq, core, input_size):
    global FULL_DATASET

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
    energy = df['Energy Reading'].item()
    return energy


def main():
    global FULL_DATASET

    FULL_DATASET = build_csv()

    initalize_r_env(f'{R_MODELS}/[PARSEC] Model Data.RData')
    # initalize_r_env(f'{R_MODELS}/Models_Predictions_v4.RData')

    for prog, input_size in PROGRAMS:

        os.chdir(DIR / 'hill-climbing')
        os.makedirs(f'{prog}_{input_size}', exist_ok=False)
        os.chdir(DIR / 'hill-climbing' / f'{prog}_{input_size}')

        for num in range(1, 1001):
            initial = fetch(prog, choice(FREQUENCIES),
                            f'N{choice(range(1, 29))}', input_size)

            search(initial, num, input_size, prog)


if __name__ == "__main__":
    main()
