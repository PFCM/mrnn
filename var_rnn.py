import tensorflow as tf
import numpy as np

import wp_test
import rnndatasets.warandpeace as data

FLAGS = tf.app.flags.FLAGS


class VarRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, noise_amt=0.0):
        self._size = num_units
        self._noise = noise_amt

    
    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            input_size = inputs.get_shape()[1].value
            weights = tf.get_variable('W',
                                      [input_size + self.state_size,
                                       self.state_size * 2])
            bias = tf.get_variable('b', [self.state_size * 2])
            args = tf.concat(1, [inputs, state])
            meanstd = tf.matmul(args, weights) + bias
            meanstd = tf.nn.tanh(meanstd)
            mean, std = tf.split(1, 2, meanstd)

            #mean = tf.nn.tanh(mean)
            #std = tf.nn.softplus(std)
            
            noise = tf.random_normal([self.state_size], stddev=0.1)
            if self._noise != 0.0:
                noise *= self._noise
            output = mean + std * noise

            # add kl to REGULARIZATION_LOSSES
            kl = 0.5 * tf.reduce_sum(mean, reduction_indices=1)
            kl += 0.5 * tf.reduce_sum(std, reduction_indices=1)
            kl -= tf.reduce_sum(tf.log(std))
            kl = tf.reduce_mean(kl)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 kl)
            
        return output, output


def inference(input_var, state_size, vocab_size, num_steps, batch_size, noise_var,
              decoder_inputs, scope):

    cell = VarRNN(state_size, noise_var)

    inputs = tf.unpack(input_var, axis=1)
    init_state = cell.zero_state(batch_size, tf.float32)
    softmax_w = tf.get_variable('softmax_w', [state_size, vocab_size])
    softmax_b = tf.get_variable('softmax_b', [vocab_size])
    outputs, state = tf.nn.seq2seq.embedding_rnn_decoder(
        inputs, init_state, cell, vocab_size, 32,
        output_projection=(softmax_w, softmax_b), scope=scope)
    logits = tf.reshape(tf.concat(1, outputs), [-1, state_size])
    logits = tf.matmul(logits, softmax_w) + softmax_b

    sample_init = cell.zero_state(1, tf.float32)
    print('got model')
    scope.reuse_variables()
    samples, _ = tf.nn.seq2seq.embedding_rnn_decoder(
        decoder_inputs, sample_init, cell, vocab_size, 32,
        output_projection=(softmax_w, softmax_b), feed_previous=True,
        scope=scope)
    samples = tf.reshape(tf.concat(1, samples), [-1, state_size])
    samples = tf.matmul(samples, softmax_w) + softmax_b
    samples = tf.argmax(samples, 1)
    samples = tf.unpack(tf.squeeze(samples))
    print('got sampling model')
    

    return logits, state, init_state, samples


def loss(logits, targets):
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.reshape(targets, [-1]))
    return tf.reduce_mean(cost)


def main(_):
    inputs = tf.placeholder(tf.int32,
                            [FLAGS.batch_size, FLAGS.num_steps])
    targets = tf.placeholder(tf.int32,
                             [FLAGS.batch_size, FLAGS.num_steps])
    vocab = data.get_vocab('char')
    inv_vocab = {b: a for a, b in vocab.items()}
    decoder_inputs = [tf.constant([vocab['<GO>']])] * 500
    noise = tf.get_variable('noise', [], trainable=False)
    
    with tf.variable_scope('rnn_model') as scope:
        logits, fstate, istate, samples = inference(
            inputs, FLAGS.width, len(vocab),
            FLAGS.num_steps, FLAGS.batch_size, noise,
            decoder_inputs, scope)

        
    av_cost = loss(logits, targets)
    reg_loss = 0.1 * tf.add_n(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES))
    train_op = wp_test.train(av_cost + reg_loss, FLAGS.learning_rate)
    print('got training op')
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(noise.assign(1.0))
    for epoch in range(FLAGS.num_epochs):
        train_iter, valid_iter, test_iter = data.get_split_iters(
            FLAGS.num_steps + 1,
            FLAGS.batch_size,
            report_progress=True,
            overlap=1)
        
        tloss = wp_test.run_epoch(sess, train_iter, istate, fstate, av_cost, train_op, inputs,
                                  targets, state_reset=0)
        sess.run(noise.assign(0.0))
        vloss = wp_test.run_epoch(sess, valid_iter, istate, fstate, av_cost,
                                  tf.no_op(), inputs, targets, state_reset=0)
        print('\n({}) Train: {}, Valid: {}'.format(epoch+1, tloss, vloss))
        sess.run(noise.assign(1.0))
        samps = sess.run(samples)
        print('\n{}\n'.format(''.join([inv_vocab[s] for s in samps])))
    

if __name__ == '__main__':
    tf.app.run()
    

