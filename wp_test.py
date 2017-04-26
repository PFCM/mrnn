"""Set up a very quick model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import sys
import os

import numpy as np
import tensorflow as tf

import rnndatasets.warandpeace as data
import mrnn

flags = tf.flags
flags.DEFINE_integer('seed', 0xface, 'graph level random seed, for'
                                     'repeatability')
flags.DEFINE_integer('sample', 0, 'If > 0, then just generate a sample'
                                  'of that length, print it to stderr and'
                                  'exit')
flags.DEFINE_string('sample_strategy', 'random', 'how to generate samples'
                                                 'one of `random`, `ml` or'
                                                 '`beam`. Beam search is not '
                                                 'implemented.')
flags.DEFINE_string('sample_seed', '"',
                    'a string to start of the rnn when sampling')
flags.DEFINE_float('learning_rate', 0.001, 'the learning rate')
flags.DEFINE_integer('num_steps', 100, 'how far to back propagate in time')
flags.DEFINE_integer('batch_size', 100, 'how many batches')
flags.DEFINE_integer('width', 256, 'how many units per layer')
flags.DEFINE_integer('reset_every', 0, 'how often to reset the state')
flags.DEFINE_integer('num_layers', 3, 'how many layers')
flags.DEFINE_integer('num_epochs', 500, 'how many times through the data')
flags.DEFINE_integer('num_chars', 1000000, 'how much of the data to use')
flags.DEFINE_boolean('profile', False, 'if true, runs the whole lot in a'
                                       'profiler')
flags.DEFINE_float('learning_rate_decay', 0.995, 'rate of linear decay of the '
                                                 'learning rate')
flags.DEFINE_integer('start_decay', 25, 'how many epochs to do before '
                                        'decaying the lr')
flags.DEFINE_float('min_lr', 1e-8, 'minimum learning rate to decay to')
flags.DEFINE_string('results_folder', '.', 'where to put the results'
                                           ' (train/valid loss)')
flags.DEFINE_string('sample_folder', 'samples', 'where to write samples')
flags.DEFINE_string('model_folder', 'models', 'where to store saved models')
flags.DEFINE_string('model_prefix', 'lstm', 'something to prepend to the name '
                                            'of the saved models')
flags.DEFINE_boolean('use_latest', False, 'whether to try load from file')
flags.DEFINE_float('dropout', 0.5, 'input dropout')
flags.DEFINE_string('weightnorm', 'none', 'how to normalise weights')
flags.DEFINE_string('nonlinearity', 'tanh', 'nonlinearity to use. should be '
                                            '"tanh" or "relu"')
flags.DEFINE_float('momentum', 0.99, 'momentum for the sgd')
flags.DEFINE_string('rec_init', 'normal',
                    'wether to use normal or identity for recurrent mat')
flags.DEFINE_string('cell', 'vanilla', 'cell to use')
flags.DEFINE_integer('rank', 128, 'rank of tensor decomposition')

FLAGS = flags.FLAGS


def inference(input_var, shape, vocab_size, num_steps,
              batch_size, return_initial_state=False, dropout=1.0,
              embedding_size=32):
    """Makes the model up to logits.

    Args:
        input_var: variable to hold the inputs.
        shape: list of layer sizes.
        vocab_size: the number of symbols in and out.
        num_steps: the length of the sequences.
        return_initial_state: whether or not to return the initial state
            variable.
        dropout: the probability of keeping the inputs. Can be a tensor,
            in which case dropout wrappers will be added and it can be changed
            on the fly. Otherwise it will be fixed, it it is 1.0 (default)
            dropout wrappers will not be added.

    Returns:
        (outputs, state) where outputs is a big tensor of output logits,
            post softmax projection (but before actual softmax)
            and state is the final state of the rnn.
    """
    # first thing we need is some kind of embedding maybe
    with tf.device('/cpu:0'):
       embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
       inputs = tf.nn.embedding_lookup(embedding, input_var)
    # inputs = tf.one_hot(input_var, vocab_size)
    #if dropout != 1.0:
    #   inputs = tf.nn.dropout(inputs, dropout)
    # set up the cells
    last_size = embedding_size
    cells = []
    # print('  weightnorm: {}'.format(FLAGS.weightnorm))
    # print('      nonlin: {}'.format(FLAGS.nonlinearity))
    for layer in shape:
        # cells.append(mrnn.IRNNCell(layer, last_size, tf.nn.elu))
        # cells.append(tf.nn.rnn_cell.BasicRNNCell(layer, last_size))
        # cells.append(tf.nn.rnn_cell.LSTMCell(layer, last_size))
        # cells.append(mrnn.SimpleRandomSparseCell(layer, last_size, .1,
        #              nonlinearity=tf.nn.tanh))
        if FLAGS.nonlinearity == 'tanh':
            nonlin = tf.nn.tanh
        elif FLAGS.nonlinearity == 'relu':
            nonlin = tf.nn.relu
        elif FLAGS.nonlinearity == 'elu':
            nonlin = tf.nn.elu
        elif FLAGS.nonlinearity == 'linear':
            nonlin = tf.identity
        else:
            raise ValueError('unknown nonlin: {}'.format(FLAGS.nonlinearity))
        if FLAGS.cell == 'vanilla':
            if FLAGS.weightnorm == 'flat-norm':
                if FLAGS.rec_init == 'identity':
                    sys.exit(0)  # just bail, we're cool
                # then we should be trying a normalised version of the whole
                # concatted matrix
                cells.append(mrnn.FlatRNNCell(layer, last_size, weightnorm=True,
                                              nonlinearity=nonlin))
            elif FLAGS.weightnorm == 'flat-none':
                if FLAGS.rec_init == 'identity':
                    sys.exit(0)
                cells.append(mrnn.FlatRNNCell(layer, last_size, weightnorm=False,
                                              nonlinearity=nonlin))
            else:
                if FLAGS.rec_init == 'normal':
                    cells.append(mrnn.VRNNCell(layer, last_size,
                                               weightnorm=FLAGS.weightnorm,
                                               nonlinearity=nonlin))
                elif FLAGS.rec_init == 'identity':
                    cells.append(mrnn.IRNNCell(layer, last_size,
                                               weightnorm=FLAGS.weightnorm,
                                               nonlinearity=nonlin))
                else:
                    raise ValueError('unknown init: {}'.format(FLAGS.rec_init))
        elif FLAGS.cell == 'simple_sparse':
            print('sparse!')
            cells.append(mrnn.SimpleRandomSparseCell2(layer, last_size,
                                                      layer**2 + last_size,
                                                      nonlin))
        elif FLAGS.cell == 'simple_cp':
            print('cp factored')
            cells.append(mrnn.SimpleCPCell(layer, last_size, layer, nonlin, True))
        elif FLAGS.cell == 'lstm':
            cells.append(tf.nn.rnn_cell.LSTMCell(layer, forget_bias=5.0))
        elif FLAGS.cell == 'simple_tt':
            print('tt')
            cells.append(mrnn.SimpleTTCell(layer, last_size, [50, 50], nonlin))
        elif FLAGS.cell == 'cp-gate':
            cells.append(mrnn.CPGateCell(layer, FLAGS.rank,
                                         candidate_nonlin=nonlin))
        elif FLAGS.cell == 'cp-gate-combined':
            cells.append(mrnn.CPGateCell(layer, FLAGS.rank, separate_pad=False,
                                         candidate_nonlin=nonlin))
        elif FLAGS.cell == 'gru':
            cells.append(tf.nn.rnn_cell.GRUCell(layer))
        elif FLAGS.cell == 'tf-vanilla':
            cells.append(tf.nn.rnn_cell.BasicRNNCell(layer))
        else:
            raise ValueError('unknown cell: {}'.format(FLAGS.cell))
        last_size = layer
    if dropout != 1.0 and 'cp-gate' not in FLAGS.cell:  # != rather than < because could be tensor
        cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
                 for cell in cells]
    if len(cells) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = cells[0]
    # make sure the inputs are an approprite list
    inputs = tf.unpack(inputs, axis=1)
    # if we need to return the initial state
    if return_initial_state:
        init_state = cell.zero_state(batch_size, tf.float32)
    else:
        init_state = None
    # and make the rnn
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=init_state,
                               dtype=tf.float32)
    # push all the outputs together
    outputs = tf.reshape(tf.concat(1, outputs), [-1, layer])
    # get softmax weights and bias
    softmax_w = tf.get_variable("softmax_w", [last_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(outputs, softmax_w) + softmax_b
    
    if return_initial_state:
        return logits, state, init_state
    return logits, state


def loss(logits, targets, vocab_size, batch_size, num_steps):
    """Gets the loss scalar.

    Args:
        logits: the raw output of the network, a big of tensors
        targets: the targets they should be getting close to
            (as an int tensor).

    Returns:
        scalar tensor representing the average cross entropy.
    """
    # print(logits.get_shape(), targets.get_shape())
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.reshape(targets, [-1]))
    cost = tf.reduce_mean(cost)
    return cost


def output_probs(logits):
    """Just pushes a softmax on the given logits, needed to sample.

    Args:
        logits: the raw activations.

    Returns:
        logits with a softmax applied.
    """
    return tf.nn.softmax(logits)


def train(cost, learning_rate, max_grad_norm=100000.0):
    """Gets the training op"""
    tvars = tf.trainable_variables()

    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.RMSPropOptimizer(learning_rate)
    g_and_v = opt.compute_gradients(cost, tvars)
    if max_grad_norm < 100000:
        grads, _ = tf.clip_by_global_norm([grad for grad, var in g_and_v],
                                          max_grad_norm)
        g_and_v = zip(grads, [var for grad, var in g_and_v])
    return opt.apply_gradients(g_and_v)


def fill_feed(batch, input_var, target_var, state, init_state):
    """Fills the feed dict with a batch (including setting up targets)"""
    inputs = batch[:-1]
    targets = batch[1:]
    # this is dumb, should just fix up the datasets
    return {
        input_var: np.array(inputs).T,
        target_var: np.array(targets).T,
        init_state: state
    }


def run_epoch(sess, data_iter, initial_state, final_state, cost, train_op,
              input_var, target_var, state_reset=FLAGS.reset_every):
    """Runs an epoch, pulling from data_iter until it is empty."""
    if state_reset > 0 and initial_state is None:
        raise ValueError("Can't reset the state without anything to reset it to")
    # get into it
    costs = 0
    steps = 0
    state = tf.zeros_like(initial_state).eval(session=sess)
    zero_state = state
    for progress, batch in data_iter:
        batch_loss, _, state = sess.run(
            [cost, train_op, final_state],
            feed_dict=fill_feed(batch,
                                input_var,
                                target_var,
                                state,
                                initial_state))
        costs += batch_loss
        steps += 1
        if state_reset != 0 and steps % state_reset == 0:
            state = zero_state
        print('\r({:.3f}) -- xent: {:.4f}'.format(progress, costs/steps),
              end='')
    print()
    return costs / steps


def gen_sample(vocab, probs, input_var, in_state_var, out_state_var,
               length=1000, seed='"'):
    """Gets a sample from the network. Uses the default session."""
    # get an initial state
    seed_list = []
    for char in seed:
        if char in vocab:
            seed_list.append(int(vocab[char]))
        else:
            seed_list.append(int(vocab['<UNK>']))
    inverse_vocab = {b: a for a, b in vocab.items()}
    sess = tf.get_default_session()
    sample = list(seed)
    state = tf.zeros_like(in_state_var).eval()

    # first initialise the rnn by running on given starting point
    for val in seed_list[:-1]:
        current_probs, state = sess.run(
            [probs, out_state_var],
            {input_var: np.array(val).reshape((1, 1)),
             in_state_var: state})

    # then go ahead and generate new things
    input_data = seed_list[-1]
    for _ in range(length):
        current_probs, state = sess.run(
            [probs, out_state_var],
            {input_var: np.array(input_data).reshape((1, 1)),
             in_state_var: state})
        if FLAGS.sample_strategy == 'random':
            try:
                input_data = np.random.multinomial(1, current_probs.flatten())
            except:
                # print('~~~~probs did not sum to one :(', file=sys.stderr)
                input_data = current_probs.flatten()
        else:
            input_data = current_probs.flatten()
        input_data = np.argmax(input_data)
        sample.append(inverse_vocab[input_data])
    return ''.join(sample)


def main(_):
    """do the stuff"""
    # first we have to get the model
    tf.set_random_seed(FLAGS.seed)
    print('\n~\n~~\n~~~\n...getting model...', end='')
    inputs = tf.placeholder(tf.int32,
                            [FLAGS.batch_size, FLAGS.num_steps])
    targets = tf.placeholder(tf.int32,
                             [FLAGS.batch_size, FLAGS.num_steps])
    inputs_1 = tf.placeholder(tf.int32, [1, 1])

    vocab = data.get_vocab('char')

    dropout = tf.get_variable('dropout', [], trainable=False)

    with tf.variable_scope('rnn_model',
                           initializer=tf.random_uniform_initializer(-0.08, 0.08)) as scope:
        full_outputs, final_state, init_state = inference(
            inputs, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), FLAGS.num_steps, FLAGS.batch_size,
            dropout=dropout, return_initial_state=True)
        av_cost = loss(full_outputs, targets, len(vocab),
                       FLAGS.batch_size, FLAGS.num_steps)
        lr_var = tf.get_variable('learning_rate', [], trainable=False)
        train_op = train(av_cost, lr_var)
        # get a one-step model for sampling
        scope.reuse_variables()
        outputs_1, state_1, init_state_1 = inference(
            inputs_1, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), 1, 1, return_initial_state=True)
        char_probs = output_probs(outputs_1)

    print('\r{:~^30}'.format('got model'))
    sess = tf.Session()
    lr = FLAGS.learning_rate  # base learning rate

    # set up a saver to save the model
    # TODO (pfcm): use a global step tensor and save it too
    saver = tf.train.Saver(tf.trainable_variables(),
                           max_to_keep=1)
    model_name = os.path.join(FLAGS.model_folder,
                              FLAGS.model_prefix)
    model_name += '({})'.format(
        '-'.join([str(FLAGS.width)] * FLAGS.num_layers))
    with sess.as_default():
        print('...initialising...', end='', flush=True)
        sess.run(tf.initialize_all_variables())
        if FLAGS.use_latest:
            checkpoint = tf.train.latest_checkpoint(FLAGS.model_folder)
            if checkpoint is None:
                print('\rIssue: no checkpoint found, but you asked to load '
                      'from one')
            else:
                saver.restore(sess, checkpoint)
                print('\r{:~^30}'.format('loaded from file'))
        else:
            print('\r{:~^30}'.format('initialised'))
        if FLAGS.sample > 0:
            print(gen_sample(vocab,
                             char_probs,
                             inputs_1,
                             init_state_1,
                             state_1,
                             length=FLAGS.sample,
                             seed=FLAGS.sample_seed),
                  file=sys.stderr)
            return
        print('\n\n{:~>30}'.format('training: '))
        tv_file = FLAGS.results_folder + '/train_valid_loss.csv'
        test_file = 'test.csv'
        print('~~~~(saving losses in {})'.format(tv_file))
        print('~~~~(saving models in {})'.format(FLAGS.model_folder))
        best_valid_loss = 1000
        best_model_path = None
        for epoch in range(FLAGS.num_epochs):
            print('~~Epoch: {}'.format(epoch+1))
            # assign the learning rate
            if epoch >= FLAGS.start_decay:
                lr *= FLAGS.learning_rate_decay
                lr = max(lr, FLAGS.min_lr)
                print('~~(new lr: {}'.format(lr))
            lr_var.assign(lr).eval()
            # data_iter = data.get_char_iter(FLAGS.num_steps+1,
            # FLAGS.batch_size,
            # report_progress=True,
            # overlap=1,
            # sequential=True)
            train_iter, valid_iter, test_iter = data.get_split_iters(
                FLAGS.num_steps+1,
                FLAGS.batch_size,
                report_progress=True,
                overlap=1)
            # make sure the dropout is set
            sess.run(dropout.assign(FLAGS.dropout))
            tloss = run_epoch(sess, train_iter, init_state, final_state, av_cost,
                              train_op, inputs, targets)
            print('~~~~Training xent: {}'.format(tloss))
            # ditch the dropout for validation purposes
            sess.run(dropout.assign(1))
            vloss = run_epoch(sess, valid_iter, init_state, final_state, av_cost,
                              tf.no_op(), inputs, targets, state_reset=0)
            print('~~~~Validation xent: {}'.format(vloss))
            if vloss < best_valid_loss:
                print('~~~~(new record)~~~~')
                best_valid_loss = vloss
                best_model_path = saver.save(
                    sess, model_name, global_step=epoch+1,
                    write_meta_graph=False)
            # write the results of this epoch
            with open(tv_file, 'a') as rf:
                rf.write('{},{},{}\n'.format(epoch, tloss, vloss))
            if (epoch+1) % 10 == 0:
                samp = gen_sample(vocab,
                                  char_probs,
                                  inputs_1,
                                  init_state_1,
                                  state_1)
                for line in samp.splitlines():
                    print('~~~~{}'.format(line))
                sample_path = os.path.join(
                    FLAGS.sample_folder, '{}.txt'.format(epoch+1))
                with open(sample_path, 'w') as f:
                    f.write(samp)
        # dropout is still 0 so let's go
        print('~'*30)
        print('Loading {}'.format(best_model_path))
        saver.restore(sess, best_model_path)
        test_loss = run_epoch(
            sess,
            test_iter,
            init_state,
            final_state,
            av_cost,
            tf.no_op(),
            inputs,
            targets,
            state_reset=0)
        print('~'*30)
        print('{:~^30}'.format('Test loss: '))
        print('{:~^30}'.format(test_loss))
        print('~'*30)
        with open(test_file, 'w') as f:
            f.write('{}'.format(test_loss))

if __name__ == '__main__':
    if FLAGS.profile:
        cProfile.run('tf.app.run()')
    else:
        tf.app.run()
