"""Very quick script to run a few tests"""
import itertools
import os
import time
import shutil
import subprocess

data_dir = 'weightnorm_tests'

weightnorm_values = ['none', 'all', 'input', 'recurrent']
nonlinearity_values = ['tanh', 'relu']

program_args = [
    'python',
    'wp_test.py',
    '--width=256',
    '--num_layers=3',
    '--batch_size=32',
    '--learning_rate=0.01',
    '--learning_rate_decay=0.95',
    '--start_decay=10',
    '--momentum=0.99',
    '--num_epochs=70',
    '--dropout=1.0',
    '--model_prefix=rnn'  # they all get separate folders so it doesn't matter
]

for wn, nonlin in itertools.product(weightnorm_values, nonlinearity_values):
    run_dir = os.path.join(
        data_dir, '{}-{}'.format(wn, nonlin))
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
        '--weightnorm=' + wn
    ]
    args = program_args + unique_args
    # print something flashy
    twidth = shutil.get_terminal_size((80, 20)).columns
    print('/' * twidth)
    print('\\' * twidth)
    print('{:/^{}}'.format('STARTING NEW RUN', twidth))
    print('{:\\^{}}'.format('({}, {})'.format(wn, nonlin), twidth))
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
    
