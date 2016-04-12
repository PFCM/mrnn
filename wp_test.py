"""Set up a very quick model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import sys

import numpy as np
import tensorflow as tf

import rnndatasets.warandpeace as data
import mrnn

flags = tf.flags
flags.DEFINE_float('learning_rate', 0.005, 'the learning rate')
flags.DEFINE_integer('num_steps', 100, 'how far to back propagate in time')
flags.DEFINE_integer('batch_size', 100, 'how many batches')
flags.DEFINE_integer('width', 100, 'how many units per layer')
flags.DEFINE_integer('num_layers', 1, 'how many layers')
flags.DEFINE_integer('num_epochs', 100, 'how many times through the data')
flags.DEFINE_integer('num_chars', 1000000, 'how much of the data to use')
flags.DEFINE_boolean('profile', False, 'if true, runs the whole lot in a'
                                       'profiler')
flags.DEFINE_float('learning_rate_decay', 0.995, 'rate of linear decay of the learning rate')
flags.DEFINE_integer('start_decay', 10, 'how many epochs to do before decaying the lr')
flags.DEFINE_float('min_lr', 1e-5, 'minimum learning rate to decay to')
flags.DEFINE_string('results_file', 'results.csv', 'where to put the results'
                                                   ' (train/valid loss, test loss)')
flags.DEFINE_string('sample_folder', 'samples', 'where to write samples')
flags.DEFINE_string('model_folder', 'models', 'where to store saved models')
flags.DEFINE_string('model_prefix', 'lstm', 'something to prepend to the name of the saved models')

FLAGS = flags.FLAGS


def inference(input_var, shape, vocab_size, num_steps,
              batch_size, return_initial_state=False):
    """Makes the model up to logits.

    Args:
        input_var: variable to hold the inputs.
        shape: list of layer sizes.
        vocab_size: the number of symbols in and out.
        num_steps: the length of the sequences.
        return_init_state: whether or not to return the initial state
            variable.

    Returns:
        (outputs, state) where outputs is a big tensor of output logits,
            post softmax projection (but before actual softmax)
            and state is the final state of the rnn.
    """
    # first thing we need is some kind of embedding
    embedding = tf.get_variable('embedding', [vocab_size, shape[0]])
    inputs = tf.nn.embedding_lookup(embedding, input_var)
    # set up the cells
    last_size = shape[0]
    cells = []
    for layer in shape:
        #cells.append(mrnn.MRNNCell(layer, last_size))
        #cells.append(tf.nn.rnn_cell.BasicRNNCell(layer))
        cells.append(tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(layer, last_size),
            input_keep_prob=0.5 if num_steps != 1 else 1.0))
        last_size = layer
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # make sure the inputs are an approprite list
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
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
    # first we have to softmax it
    cost = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([batch_size * num_steps])])
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


def train(cost, learning_rate, max_grad_norm=10.0):
    """Gets the training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    #opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9,
                                 beta2=0.95, epsilon=1e-6)
    return opt.apply_gradients(zip(grads, tvars))


def fill_feed(batch, input_var, target_var):
    """Fills the feed dict with a batch (including setting up targets)"""
    inputs = batch[:-1]
    targets = batch[1:]
    return {
        input_var: inputs,
        target_var: targets
    }


def run_epoch(sess, data_iter, initial_state, final_state, cost, train_op,
              input_var, target_var):
    """Runs an epoch, pulling from data_iter until it is empty."""

    # get into it
    costs = 0
    steps = 0
    for progress, batch in data_iter:
        batch_loss, _ = sess.run([cost, train_op],
                                 feed_dict=fill_feed(batch,
                                                     input_var,
                                                     target_var))
        costs += batch_loss
        steps += 1
        print('\r({:.3f}) -- xent: {:.4f}'.format(progress, costs/steps), end='')
    print()
    return costs / steps


def gen_sample(vocab, probs, input_var, in_state_var, out_state_var, length=1000):
    """Gets a sample from the network. Uses the default session."""
    # get an initial state
    input_data = int(vocab['"'])  # first letter of the book
    inverse_vocab = {b: a for a, b in vocab.items()}
    sess = tf.get_default_session()
    sample = ['"']
    state = tf.zeros_like(in_state_var).eval()
    for _ in range(length):
        current_probs, state = sess.run(
            [probs, out_state_var],
            {input_var: np.array(input_data).reshape((1, 1)),
             in_state_var: state})
        try:
            input_data = np.random.multinomial(1, current_probs.flatten())
        except:
            #print('~~~~probs did not sum to one :(', file=sys.stderr)
            input_data = current_probs.flatten()
        input_data = np.argmax(input_data)
        sample.append(inverse_vocab[input_data])
    return ''.join(sample)       


def main(_):
    """do the stuff"""
    # first we have to get the model
    print('...getting model...', end='')
    inputs = tf.placeholder(tf.int32,
                            [FLAGS.batch_size, FLAGS.num_steps])
    targets = tf.placeholder(tf.int32,
                             [FLAGS.batch_size, FLAGS.num_steps])
    inputs_1 = tf.placeholder(tf.int32, [1, 1])
    
    vocab = data.get_vocab('char')

    with tf.variable_scope('rnn_model') as scope:
        full_outputs, final_state = inference(
            inputs, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), FLAGS.num_steps, FLAGS.batch_size)
        av_cost = loss(full_outputs, targets, len(vocab),
                       FLAGS.batch_size, FLAGS.num_steps)
        lr_var = tf.get_variable('learning_rate', [])
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
    with sess.as_default():
        print('...initialising...', end='')
        sess.run(tf.initialize_all_variables())
        print('\r{:~^30}'.format('initialised'))
        print('\n\n{:~>30}'.format('training: '))
        tv_file = 'train_valid_' + FLAGS.results_file
        print('~~~~(saving losses in {})'.format(tv_file))
        print('~~~~(saving models in {})'.format(FLAGS.model_folder))
        for epoch in range(FLAGS.num_epochs):
            print('~~Epoch: {}'.format(epoch+1))
            # assign the learning rate
            if epoch >= FLAGS.start_decay:
                lr *= FLAGS.learning_rate_decay
                lr = max(lr, FLAGS.min_lr)
                print('~~(new lr: {}'.format(lr))
            lr_var.assign(lr).eval()
            #data_iter = data.get_char_iter(FLAGS.num_steps+1,
            #FLAGS.batch_size,
            #report_progress=True,
            #overlap=1,
            #sequential=True)
            train_iter, valid_iter, test_iter = data.get_split_iters(
                FLAGS.num_steps+1,
                FLAGS.batch_size,
                report_progress=True)
            tloss = run_epoch(sess, train_iter, None, final_state, av_cost, train_op,
                              inputs, targets)
            print('~~~~Training xent: {}'.format(tloss))
            vloss = run_epoch(sess, valid_iter, None, final_state, av_cost, tf.no_op,
                              inputs, targets)
            print('~~~~Validation xent: {}'.format(vloss))
            print('BUT NOT ACTUALLY BECAUSE YOU LEFT THE DROPOUT ON DIDN\'T YOU')
            # write the results of this epoch
            with open(FLAGS.results_file, 'a') as rf:
                rf.write('{},{},{}\n'.format(epoch, tloss, vloss))
            if (epoch+1)%10 == 0:
                samp = gen_sample(vocab, char_probs, inputs_1, init_state_1, state_1)
                for line in samp.splitlines():
                    print('~~~~{}'.format(line))
                sample_path = os.path.join(FLAGS.sample_folder, '{}.txt'.format(epoch+1))
                with open(sample_path, 'w') as f:
                    f.write(samp)
                print('SHOULD SAVE MODEL HERE')
        print('AND THIS IS WHERE YOU SHOULD RUN IT ON THE TEST DATA')

if __name__ == '__main__':
    if FLAGS.profile:
        cProfile.run('tf.app.run()')
    else:
        tf.app.run()
