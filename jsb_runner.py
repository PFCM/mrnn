"""Run a grid search over RNNs on JSB. Probably sequentially? Might be able
to do a couple at once on one machine even, they're pretty lightweight."""
import itertools
import os
import time
import shutil
import subprocess

results_dir = 'jsb_gridsearch'

cell_values = ['vanilla', 'cp-gate', 'gru', 'lstm', 'simple-cp']
lr_values = ['0.1', '0.01', '0.001']
batch_sizes = ['4', '8', '16', '32']
sequence_lengths = ['10', '25', '35', '50', '75', '100']

grid_iter = itertools.product(cell_values, lr_values, batch_sizes,
                              sequence_lengths)

for cell, lr, batch_size, seq_len in grid_iter:
    run_dir = os.path.join(
        results_dir, '{}-{}-{}-{}'.format(cell, lr, batch_size, seq_len))
    os.makedirs(run_dir, exist_ok=True)
    # get ready to run
    twidth = shutil.get_terminal_size((80, 20)).columns
    print('^*^' * (twidth // 3))
    print('{:*^{}}'.format('{}, lr: {}, bs: {}, sl: {}'.format(
        cell, lr, batch_size, seq_len), twidth))
    args = ['python',
            'jsb_test.py',
            '--width=50',
            '--num_layers=1',
            '--num_epochs=250',
            '--cell=' + cell,
            '--learning_rate=' + lr,
            '--batch_size=' + batch_size,
            '--sequence_length=' + seq_len,
            '--results_dir=' + run_dir]
    start = time.time()
    subprocess.run(args, check=True)
    end = time.time()
    print('\n\n')
    print('>*>' * (twidth // 3))
    print('<*<' * (twidth // 3))
    print('{:*^{}}\n'.format('({}s)'.format(end-start), twidth))
