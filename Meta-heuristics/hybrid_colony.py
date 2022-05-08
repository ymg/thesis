#!/usr/bin/env python3
from rdata_wrapper_2 import initalize_r_env, get_model_prediction
import csv
import glob
import os
import random
import re
import string
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas

sys.path.append("model_sampling/py_sde_perf")

lowercase = string.ascii_lowercase

FULL_DATASET = None
PROGRAM = ''
INPUT_SIZE = ''

freq_list = [1.2, 1.4, 1.8, 2.1, 2.6]
core_list = list(range(2, 29))

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


def hybrid_ant_colony(file_id):
    # visisted states in all nodes
    state_history = []

    # constant for delta random accept factor
    delta_accepted_threshold = 0.5

    # initial delta value
    delta = 5

    # dictionary to seed inital nodes in the colony, contains randomly generated nodes
    random_nodes = defaultdict(tuple)

    f = open(f'sample_{file_id}.csv', 'w')

    write = csv.writer(f)

    if FULL_DATASET:

        for i in range(21):
            new_node = fetch(PROGRAM, random.choice(FREQUENCIES),
                             f'N{random.choice(range(1, 29))}', INPUT_SIZE)
            random_nodes[i] = (1, new_node)

        write.writerow(random_nodes[0][1][4].columns.values.tolist())

        ############################
        # running hybrid colony loop
        for node, val in random_nodes.items():

            initial_node = val[1]
            best_state = initial_node
            state_history.append(initial_node)

            for _ in range(3):

                rng_core = random.choice(core_list)
                rng_freq = random.choice(freq_list)
                next_state = fetch(PROGRAM, rng_freq, rng_core, INPUT_SIZE)

                new_delta = random.uniform(0, 1)

                next_state_energy = next_state[3]
                best_state_energy = best_state[3]

                if next_state_energy < best_state_energy:
                    best_state = next_state
                    delta -= new_delta
                    state_history.append(next_state)

                elif new_delta >= delta_accepted_threshold:
                    delta += new_delta
                    best_state = next_state
                    state_history.append(next_state)

                else:
                    state_history.append(best_state)

    lowest_energy_df = None
    for u in state_history:
        if not lowest_energy_df:
            lowest_energy_df = u

        if u[3] < lowest_energy_df[3]:
            lowest_energy_df = u

        write.writerow(u[4].values.flatten().tolist())

    print(f'List size: {len(state_history)}', file=f)
    print(
        f'Search lowest is: {lowest_energy_df[1]}, {lowest_energy_df[2]}, {get_cores(lowest_energy_df[4])}, {get_energy(lowest_energy_df[4])} Joules', file=f)
    print(f'Estimated lowest is: {lowest_energy_df[3]} Joules', file=f)

    f.close()


def main():
    global FULL_DATASET, PROGRAM, INPUT_SIZE

    initalize_r_env(f'{R_MODELS}/[PARSEC] Model Data.RData')
    # initalize_r_env(f'{R_MODELS}/Models_Predictions_v4.RData')

    for p, i in PROGRAMS:
        PROGRAM = p
        INPUT_SIZE = i

        FULL_DATASET = build_csv()

        os.chdir(DIR / 'ant-colony')

        os.makedirs(f'{p}_{i}', exist_ok=False)

        os.chdir(DIR / 'ant-colony' / f'{p}_{i}')

        for __i in range(1, 1001):
            hybrid_ant_colony(__i)


if __name__ == "__main__":
    main()
