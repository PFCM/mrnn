"""Run a grid search over RNNs on JSB. Probably sequentially? Might be able
to do a couple at once on one machine even, they're pretty lightweight."""
import itertools
import os
import time
import shutil
import subprocess

import numpy as np


def get_width(cell, rank, params=20000):
    """Gets the number of units with the closest number of parameters to
    `params` for a given type of cell.
    
    Assumes inputs are size 55 and includes biases and the output layer
    """
    if cell == 'vanilla':
        # 2*55*hidden + hidden*hidden + hidden + 55 = params
        # solve for hidden
        result = np.round(np.max(np.roots(np.array([1, 111, 55-params]))))
    elif cell == 'gru':
        # (55*hidden + hidden*hidden + hidden) * 3 + 55*hidden + 55
        # 4*55*hidden + 3*hidden*hidden + 3*hidden + 55 
        result = np.round(np.max(np.roots(np.array([3, 56*4+3, 55-params]))))
    elif cell == 'lstm':
        # (55*hidden + hidden*hidden +hidden) * 4 + 55*hidden + 55
        # = 5*55*hidden + 4*hidden*hidden + 4*hidden + 55
        result = np.round(np.max(np.roots(np.array([4, 55*5+4, 55-params]))))
    elif cell == 'simple_cp':
        # 55*hidden + hidden*hidden + 2*rank*hidden + rank*55 + hidden + 55*hidden + 55
        quad, lin, constant = 1, 111, 55-params
        if rank == 'one':
            lin += 2
            constant += 55
        if rank == 'half':
            quad += 1
            lin += 55/2
        if rank == 'full':
            quad += 2
            lin += 55
        if rank == 'double':
            quad += 4
            lin += 55*2
        result = np.round(np.max(np.roots(np.array([quad, lin, constant]))))
    elif cell == 'cp-gate':
        # 3*55*hidden + hidden*hidden + 2*rank*hidden + rank*55 + hidden + 55
        quad, lin, constant = 1, 3*55+1, 55-params
        if rank == 'one':
            lin += 2
            constant += 55
        if rank == 'half':
            quad += 1
            lin += 55/2
        if rank == 'full':
            quad += 2
            lin += 55
        if rank == 'double':
            quad += 4
            lin += 55*2
        result = np.round(np.max(np.roots(np.array([quad, lin, constant]))))
    elif cell == 'cp-gate-combined':
        # 2*55*hidden + hidden + rank*hidden + rank*(hidden+1) + rank*(55+1) + 55
        # = (2*55+1)*hidden + rank*hidden + rank*hidden + rank + 55*rank + 55 + 55
        # = (2*55+1)*hidden + 2*rank*hidden + 56*rank + 110
        quad, lin, constant = 0, 111, 110-params
        if rank == 'one':
            lin += 2
            constant += 57
        if rank == 'half':
            quad += 1
            lin += 57/2
        if rank == 'full':
            quad += 2
            lin += 57
        if rank == 'double':
            quad += 4
            lin += 57*2
        result = np.round(np.max(np.roots(np.array([quad, lin, constant]))))

    return int(result)


def get_rank(width, rank):
    if rank == 'one':
        return 1
    if rank == 'half':
        return width//2
    if rank == 'full':
        return width
    if rank == 'double':
        return width*2


results_dir = 'jsb_gridsearch_fair'

cell_values = [
    'vanilla',
    'cp-gate',
    'gru',
    'lstm',
    'simple_cp',
    'cp-gate-combined']
lr_values = ['0.1', '0.01', '0.001']
batch_sizes = ['8']
sequence_lengths = ['75', '100']
ranks = ['one', 'half', 'full', 'double']

grid_iter = itertools.product(cell_values, lr_values, batch_sizes,
                              sequence_lengths, ranks)

for cell, lr, batch_size, seq_len, rank in grid_iter:
    if 'cp' not in cell and rank != 'one':
        continue
    run_dir = os.path.join(
        results_dir, '{}-{}-{}-{}-rank{}'.format(cell, lr, batch_size, seq_len, rank))
    os.makedirs(run_dir, exist_ok=True)
    # get ready to run
    twidth = shutil.get_terminal_size((80, 20)).columns
    width = get_width(cell, rank)
    rank = get_rank(width, rank)
    print('^*^' * (twidth // 3))
    print('{:*^{}}'.format('{}, lr: {}, bs: {}, sl: {}, (w: {},  r: {})'.format(
        cell, lr, batch_size, seq_len, width, rank), twidth))

    args = ['python',
            'jsb_test.py',
            '--width={}'.format(width),
            '--num_layers=1',
            '--num_epochs=200',
            '--rank={}'.format(rank),
            '--cell=' + cell,
            '--learning_rate=' + lr,
            '--batch_size=' + batch_size,
            '--sequence_length=' + seq_len,
            '--results_dir=' + run_dir]
    start = time.time()
    with subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1) as p:
        stdout = []
        for line in p.stdout:
            print(line.decode(), end='', flush=True)
            stdout.append(line.decode())
    end = time.time()
    with open(os.path.join(run_dir, 'stdout.txt'), 'w') as fp:
        fp.write(''.join(stdout))
    print('\n\n')
    print('>*>' * (twidth // 3))
    print('<*<' * (twidth // 3))
    print('{:*^{}}\n'.format('({}s)'.format(end-start), twidth))
