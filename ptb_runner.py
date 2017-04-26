"""Grid search over rank on PTB"""
import itertools
import os
import subprocess
import time

cells = [
    'cp-gate',
    'cp-gate-combined',
    'cp-gate-linear',
    'cp-gate-combined-linear',
    'gru',
    'lstm',
    'vanilla']

learning_rates = ['0.02', '0.01', '0.005']
ranks = ['1', '8', '32', '64', '128', '256']

grid_iter = itertools.product(cells, learning_rates, ranks)

for cell, lr, rank in grid_iter:
    if 'cp' not in cell and rank != '1':
        continue

    print('>%>|' * 20)
    print('cell: {}, lr: {}, rank: {}'.format(cell, lr, rank))

    results_dir = os.path.join('ptb_grid_bs100_sl50', cell,
                               'lr-{}'.format(lr),
                               'rank-{}'.format(rank))
    os.makedirs(results_dir)
    args = [
        'python',
        'ptb_test.py',
        '--width=128',
        '--num_layers=1',
        '--num_epochs=10',
        '--rank={}'.format(rank),
        '--batch_size=100',
        '--sequence_length=50',
        '--learning_rate={}'.format(lr),
        '--cell={}'.format(cell),
        '--results_dir={}'.format(results_dir)]

    start = time.time()
    with subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1) as p:
        stdout = []
        for line in p.stdout:
            print(line.decode(), end='', flush=True)
            stdout.append(line.decode())
    end = time.time()
    with open(os.path.join(results_dir, 'stdout.txt'), 'w') as fp:
        fp.write(''.join(stdout))
    print('\n\n')
    print('<*>><'*20)
    print('({}s)'.format(end-start))
