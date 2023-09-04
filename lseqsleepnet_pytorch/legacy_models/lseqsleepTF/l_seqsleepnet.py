import tensorflow as tf
from nn_basic_layers import *
from filterbank_shape import FilterbankShape
from ops import *

class L_SeqSleepNet(object):

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, self.config.nseg, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_ndim, self.config.nchannel], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.nseg, self.config.epoch_seq_len, self.config.nclass], name="input_y")

        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization
        self.seq_frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.chunk_seq_len = tf.placeholder(tf.int32, [None])  # for the dynamic RNN
        self.filtershape = FilterbankShape()


        x = tf.reshape(self.input_x, [-1, self.config.seq_ndim, self.config.nchannel])
        processed_x = self.preprocessing(x)
        # [-1, nfilters*nchannel]

        processed_x = tf.reshape(processed_x, [-1, self.config.seq_frame_seq_len, self.config.seq_nfilter*self.config.nchannel])
        epoch_x = self.epoch_encoder(processed_x)
        # [-1, nhidden1*2]

        epoch_x = tf.reshape(epoch_x, [-1, self.config.nseg, self.config.epoch_seq_len, self.config.seq_nhidden1*2])

        seq_x = self.dual_sequence_encoder(epoch_x, self.config.dualrnn_blocks)

        with tf.variable_scope("output_layer"):
            X_out = tf.reshape(seq_x, [-1, self.config.seq_nhidden1*2])
            fc1 = fc(X_out, self.config.seq_nhidden1*2, self.config.fc_size, name="fc1", relu=True)
            fc1 = dropout(fc1, self.dropout_rnn)
            fc2 = fc(fc1, self.config.fc_size, self.config.fc_size, name="fc2", relu=True)
            fc2 = dropout(fc2, self.dropout_rnn)
            self.score = fc(fc2, self.config.fc_size, self.config.nclass, name="output", relu=False)
            self.prediction = tf.argmax(self.score, 1, name="pred")
            self.score = tf.reshape(self.score, [-1, self.config.nseg, self.config.epoch_seq_len, self.config.nclass])
            self.prediction = tf.reshape(self.prediction, [-1, self.config.nseg, self.config.epoch_seq_len])

        # calculate sequence cross-entropy output loss
        self.output_loss = 0.0
        with tf.name_scope("output-loss"):
            y = tf.reshape(self.input_y, [-1, self.config.nclass])
            logit = tf.reshape(self.score, [-1, self.config.nclass])
            self.output_loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit), axis=[0])
            self.output_loss /= (self.config.nseg * self.config.epoch_seq_len)  # average over sequence length

        # add on regularization
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy at each time index of the input sequence
        with tf.name_scope("accuracy"):
            y = tf.reshape(self.input_y, [-1, self.config.nseg * self.config.epoch_seq_len, self.config.nclass])
            yhat = tf.reshape(self.prediction, [-1, self.config.nseg * self.config.epoch_seq_len])
            for i in range(self.config.nseg * self.config.epoch_seq_len):
                correct_prediction_i = tf.equal(yhat[:, i], tf.argmax(tf.squeeze(y[:, i, :]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

    def preprocessing(self, input):
        # [-1, ndim, nchannel]
        # triangular filterbank
        Wbl = tf.constant(self.filtershape.lin_tri_filter_shape(nfilt=self.config.seq_nfilter,
                                                                nfft=self.config.seq_nfft,
                                                                samplerate=self.config.seq_samplerate,
                                                                lowfreq=self.config.seq_lowfreq,
                                                                highfreq=self.config.seq_highfreq),
                          dtype=tf.float32,
                          name="W-filter-shape-eeg")

        with tf.variable_scope("seq_filterbank-layer-eeg", reuse=tf.AUTO_REUSE):
            # Temporarily crush the feature_mat's dimensions
            Xeeg = tf.reshape(tf.squeeze(input[:, :, 0]), [-1, self.config.seq_ndim])
            # first filter bank layer
            Weeg = tf.get_variable('Weeg', shape=[self.config.seq_ndim, self.config.seq_nfilter], initializer=tf.random_normal_initializer())
            # non-negative constraints
            Weeg = tf.sigmoid(Weeg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            Wfb_eeg = tf.multiply(Weeg, Wbl)
            HWeeg = tf.matmul(Xeeg, Wfb_eeg)  # filtering
            #HWeeg = tf.reshape(HWeeg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if (self.config.nchannel > 1):
            with tf.variable_scope("seq_filterbank-layer-eog", reuse=tf.AUTO_REUSE):
                # Temporarily crush the feature_mat's dimensions
                Xeog = tf.reshape(tf.squeeze(input[:, :, 1]), [-1, self.config.seq_ndim])
                # first filter bank layer
                # self.Weog = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                Weog = tf.get_variable('Weog', shape=[self.config.seq_ndim, self.config.seq_nfilter],initializer=tf.random_normal_initializer())
                # non-negative constraints
                Weog = tf.sigmoid(Weog)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                Wfb_eog = tf.multiply(Weog, Wbl)
                HWeog = tf.matmul(Xeog, Wfb_eog)  # filtering
                #HWeog = tf.reshape(HWeog, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if (self.config.nchannel > 2):
            with tf.variable_scope("seq_filterbank-layer-emg", reuse=tf.AUTO_REUSE):
                # Temporarily crush the feature_mat's dimensions
                Xemg = tf.reshape(tf.squeeze(input[:, :, 2]), [-1, self.config.seq_ndim])
                # first filter bank layer
                # self.Wemg = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                Wemg = tf.get_variable('Wemg', shape=[self.config.seq_ndim, self.config.seq_nfilter], initializer=tf.random_normal_initializer())
                # non-negative constraints
                Wemg = tf.sigmoid(Wemg)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                Wfb_emg = tf.multiply(Wemg, Wbl)
                HWemg = tf.matmul(Xemg, Wfb_emg)  # filtering
                #HWemg = tf.reshape(HWemg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if (self.config.nchannel > 2):
            X2 = tf.concat([HWeeg, HWeog, HWemg], axis=1)
        elif (self.config.nchannel > 1):
            X2 = tf.concat([HWeeg, HWeog], axis=1)
        else:
            X2 = HWeeg

        return X2
        #X2 = tf.reshape(X2, [-1, self.config.seq_frame_seq_len, self.config.seq_nfilter * self.config.nchannel])

    def epoch_encoder(self, input):
        # [-1, frame_seq_len, dim]
        # bidirectional frame-level recurrent layer
        with tf.variable_scope("seq_frame_rnn_layer", reuse=tf.AUTO_REUSE) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(self.config.seq_nhidden1,
                                                                    self.config.seq_nlayer1,
                                                                    seq_len=self.config.seq_frame_seq_len,
                                                                    is_training=self.istraining,
                                                                    input_keep_prob=self.dropout_rnn,
                                                                    output_keep_prob=self.dropout_rnn)
            rnn_out1, _ = bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input, self.seq_frame_seq_len, scope=scope)
            print(rnn_out1.get_shape())
            # [-1, frame_seq_len, nhidden*2]

        with tf.variable_scope("frame_attention_layer", reuse=tf.AUTO_REUSE):
            frame_attention_out1, _ = attention(rnn_out1, self.config.seq_attention_size1)
            print(frame_attention_out1.get_shape())
            #seq_e_rnn_input = tf.reshape(frame_attention_out1, [-1, self.config.epoch_seq_len, self.config.seq_nhidden1 * 2])
            # [-1, nhidden*2]
        return frame_attention_out1

    def residual_rnn(self, input, seq_len, in_dropout=1.0, out_dropout=1.0, name='rnn_res'):
        #[-1, nseq, dim]
        _, nseq, dim = input.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(self.config.seq_nhidden2,
                                                                    self.config.seq_nlayer2,
                                                                    seq_len = nseq,
                                                                    is_training=self.istraining,
                                                                    input_keep_prob=in_dropout,
                                                                    output_keep_prob=out_dropout)
            rnn_out, _ = bidirectional_recurrent_layer_output_new(fw_cell, bw_cell, input, seq_len, scope=scope)
            # linear projection
            rnn_out = fc(tf.reshape(rnn_out, [-1, self.config.seq_nhidden2*2]),
                         self.config.seq_nhidden2*2, dim, name="fc", relu=False)
            rnn_out = tf.contrib.layers.layer_norm(rnn_out)
            #rnn_out = dropout(rnn_out, out_dropout)
            # residual
            rnn_out = tf.reshape(rnn_out, [-1, nseq, dim]) + input

            return rnn_out

    def dual_sequence_encoder(self, input, N):
        _, nchunk, chunk_len, dim = input.get_shape().as_list()
        # (Batch, B, K, Features)
        for n in range(N):
            input = tf.reshape(input, [-1, chunk_len, dim])
            # (Batch*B, Sequence, Features)
            rnn1_out = self.residual_rnn(input, self.epoch_seq_len,
                                         in_dropout = 1.0 if n == 0 else self.dropout_rnn,
                                         out_dropout = self.dropout_rnn,
                                         name = 'intra_chunk_rnn_' + str(n+1))
            rnn1_out = tf.reshape(rnn1_out, [-1, nchunk, chunk_len, dim]) # [-1, nchunk, nseq, nhidden2*2]
            rnn1_out = tf.transpose(rnn1_out, perm=[0, 2, 1, 3]) # [-1, nseq, nchunk, nhidden2*2]
            rnn1_out = tf.reshape(rnn1_out, [-1, nchunk, dim])

            rnn2_out = self.residual_rnn(rnn1_out, self.chunk_seq_len,
                                         in_dropout = self.dropout_rnn,
                                         out_dropout = self.dropout_rnn,
                                         name = 'inter_chunk_rnn_' + str(n+1))
            rnn2_out = tf.reshape(rnn2_out, [-1, chunk_len, nchunk, dim])  # [-1, nchunk, nseq, nhidden2*2]
            rnn2_out = tf.transpose(rnn2_out, perm=[0, 2, 1, 3])  # [-1, nseq, nchunk, nhidden2*2]
            rnn2_out = tf.reshape(rnn2_out, [-1, chunk_len, dim])
            input = rnn2_out

        input = tf.reshape(input, [-1, nchunk, chunk_len, dim])
        input = dropout(input, self.dropout_rnn)
        return input


