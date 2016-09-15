"""Very quick script to run a few tests"""
import itertools
import os
import time
import shutil
import subprocess
# test comment
data_dir = 'wp_grid'

weightnorm_values = []#'full', 'input', 'recurrent', 'none',
                     #'flat-norm']
nonlinearity_values = ['relu', 'linear']
#inits = ['normal', 'identity']
ranks = ['1', '8', '32', '128', '256']
lr_vals = ['0.001']
cells = ['cp-gate-combined', 'cp-gate']
dropout = ['0.5', '0.8', '0.9']

program_args = [
    'python',
    'wp_test.py',
    '--width=128',
    '--num_layers=1',
    '--batch_size=64',
    '--learning_rate_decay=0.95',
    '--start_decay=10000',
    '--momentum=0.95',
    '--num_epochs=50',
    '--dropout=0.75',
    '--model_prefix=rnn'  # they all get separate folders so it doesn't matter
]

grid_iter = itertools.product(cells, nonlinearity_values,
                              ranks, lr_vals)

for cell, nonlin, rank, lr in grid_iter:
    run_dir = os.path.join(
        data_dir, '{}-{}'.format(cell, nonlin))
    run_dir = os.path.join(
        run_dir, 'rank-{}'.format(rank))
    model_dir = os.path.join(run_dir, 'models')
    sample_dir = os.path.join(run_dir, 'samples')
    # make directories if necessary
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    # marshal the arguments
    unique_args = [
        '--results_folder=' + run_dir,
        '--model_folder=' + model_dir,
        '--sample_folder=' + sample_dir,
        '--nonlinearity=' + nonlin,
        '--learning_rate=' + lr,
        '--rank=' + rank,
        '--cell=' + cell
    ]
    args = program_args + unique_args
    # print something flashy
    twidth = shutil.get_terminal_size((80, 20)).columns
    print('/' * twidth)
    print('\\' * twidth)
    print('{:/^{}}'.format('STARTING NEW RUN', twidth))
    print('{:\\^{}}'.format('({}, {}, {})'.format(
        cell, nonlin, rank), twidth))
    print('{:/^{}}'.format(run_dir, twidth))
    print('/' * twidth)
    print('\\' * twidth)
    # and run, letting an except propagate if it fails
    start = time.time()
    subprocess.run(args, check=True)
    end = time.time()
    print('\n\n')
    print('/' * twidth)
    print('\\' * twidth)
    print('{:/^{}}'.format('done', twidth))
    print('{:\\^{}}'.format('({}s)'.format(end-start), twidth))
    print('/' * twidth)
    print('\\' * twidth)
