import os
import numpy as np
import random
import itertools
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import pretty_midi
import time
from tensorflow.contrib.rnn import LSTMCell, LSTMBlockCell
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class violin_fingering_model(object):
    def __init__(self):
        self.pitch_for_invalid_note = 101
        self.lowest_pitch = 55
        self.n_p_classes = 46 + 1 # number of pitch classes, pitch range =  55 to 100, pitch_for_invalid_note = 101
        self.n_b_classes = 7 # number of beat_type classes, {'', '1th', '2th', '4th',  '8th',  '16th', '32th'}
        self.n_str_classes = 5 # number of string classes, {'', G', 'D', 'A', 'E'}
        self.n_pos_classes = 13 # number of position classes, {'', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}
        self.n_fin_classes = 5 # number of finger classes, {'' & '0', '1', '2', '3', '4'}
        self.n_steps = 32 # number of note events for an input sequence
        self.embedding_size = 100
        self.hidden_size = 100
        self.batch_size = 26
        self.INFIN = 1e12 # infinite number
        self.n_in_succession = 10
        self.initial_learning_rate = 1e-3
        self.n_epochs = 100
        self.drop = 0.3
        self.dataset_dir = "./TNUA_violin_fingering_dataset"
        self.save_dir = './model'
        self.model_dir = './model/violin_fingering_estimation.ckpt'

    def load_data(self):
        print("Load training data...")
        files = [x for x in os.listdir(self.dataset_dir) if x.endswith('csv')]
        corpus = {}
        for file in files:
            with open(self.dataset_dir + '/' + file) as f:
                corpus[file] = np.genfromtxt(f, delimiter=',', names=True, dtype=[('int'), ('float'), ('float'), ('int'), ('int'), ('int'), ('int')])
        return corpus

    def segment_corpus(self, corpus):
        def _segment_sequence(sequence, max_len):
            n_pad = max_len - (len(sequence) % max_len) if len(sequence) % max_len > 0 else 0
            dt = sequence.dtype
            paddings = np.array([(self.pitch_for_invalid_note, -1, 0, 0, 0, 0, 0) for _ in range(n_pad)], dtype=dt)
            sequence_padded = np.concatenate([sequence, paddings])
            segments = np.reshape(sequence_padded, newshape=[-1, max_len])
            valid_lens = [max_len for _ in range(segments.shape[0] - 1)] + [(len(sequence) % max_len)] if len( sequence) % max_len > 0 else [max_len for _ in range(segments.shape[0])]
            return segments, valid_lens
        corpus_seg = {}  # {key: {segments: 2d_array, lens: len_list}}
        for key, sequence in corpus.items():
            corpus_seg[key] = {}
            segments, valid_lends = _segment_sequence(sequence, max_len=self.n_steps)
            corpus_seg[key]['segments'] = segments
            corpus_seg[key]['lens'] = valid_lends
        print('total number of segments =', sum([v['segments'].shape[0] for v in corpus_seg.values()]))
        return corpus_seg

    def create_training_and_testing_sets(self, corpus):
        corpus_vio1 = {k: v for k, v in corpus.items() if 'vio1_' in k}  # only use vio1
        training_key_list = [key for key in corpus_vio1.keys() if any(x in key for x in ['bach', 'mozart', 'beeth', 'mend', 'flower', 'wind'])]
        training_data = [v for k, v in corpus_vio1.items() if k in training_key_list]
        testing_data = [v for k, v in corpus_vio1.items() if k not in training_key_list]

        training_segments = np.concatenate([x['segments'] for x in training_data], axis=0)
        training_lens = np.array(list(itertools.chain.from_iterable([x['lens'] for x in training_data])))
        testing_segments = np.concatenate([x['segments'] for x in testing_data], axis=0)
        testing_lens = np.array(list(itertools.chain.from_iterable([x['lens'] for x in testing_data])))
        print('shape of training data =', training_segments.shape)
        print('shape of testing data =', testing_segments.shape)

        X = {'train': {'pitch': training_segments['pitch'],
                       'start': training_segments['start'],
                       'duration': training_segments['duration'],
                       'beat_type': training_segments['beat_type'],
                       'lens': training_lens},
             'test': {'pitch': testing_segments['pitch'],
                      'start': testing_segments['start'],
                      'duration': testing_segments['duration'],
                      'beat_type': testing_segments['beat_type'],
                      'lens': testing_lens}}

        Y = {'train': {'string': training_segments['string'],
                       'position': training_segments['position'],
                       'finger': training_segments['finger']},
             'test': {'string': testing_segments['string'],
                      'position': testing_segments['position'],
                      'finger': testing_segments['finger']}
             }
        return X, Y

    def normalize(self, inputs, epsilon=1e-6, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(tf.squared_difference(inputs, mean), axis=[-1], keepdims=True)
            normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)

            beta = tf.get_variable("beta_bias", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            outputs = gamma * normalized + beta
        return outputs

    def BLSTM(self, x_p, x_s, x_d, x_b, x_len, dropout, activation=tf.nn.tanh):
        '''p=pitch, s=start, d=duration, b=beat_type'''
        with tf.name_scope('Input_embedding'):
            x_p_onehot = tf.one_hot(x_p - self.lowest_pitch, depth=self.n_p_classes)
            x_b_onehot = tf.one_hot(x_b, depth=self.n_b_classes)
            input = tf.concat([x_p_onehot, x_s[:, :, None], x_d[:, :, None], x_b_onehot], axis=2)
            input_embedded = tf.layers.dense(input, self.embedding_size)
            input_embedded = self.normalize(input_embedded, scope='input_ln')
            input_embedded = activation(input_embedded)
            input_embedded = tf.nn.dropout(input_embedded, keep_prob=1-dropout)

        with tf.name_scope('BLSTM_cells'):
            cell_fw = LSTMBlockCell(num_units=self.hidden_size, name='cell_fw') # LSTMCell(num_units=hidden_size, name='cell_fw')
            cell_bw = LSTMBlockCell(num_units=self.hidden_size, name='cell_bw') # LSTMCell(num_units=hidden_size, name='cell_bw')

        with tf.name_scope('RNN'):
            # bi-LSTM
            (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                             cell_bw=cell_bw,
                                                                             inputs=input_embedded,
                                                                             sequence_length=x_len,
                                                                             dtype=tf.float32,
                                                                             time_major=False)
            hidden_states = tf.concat((output_fw, output_bw), axis=-1)

            hidden_states = self.normalize(hidden_states, scope='hidden_ln')
            hidden_states = activation(hidden_states)
            hidden_states = tf.nn.dropout(hidden_states, keep_prob=1-dropout)

        with tf.name_scope('Output'):
            s_logits = tf.layers.dense(hidden_states, self.n_str_classes, name='string_out')
            p_logits = tf.layers.dense(hidden_states, self.n_pos_classes, name='position_out')

        return s_logits, p_logits

    def create_str_mask(self, pitches):
        '''n_str_classes = 5 # {'', G', 'D', 'A', 'E'}'''
        null_str_mask = tf.zeros_like(pitches, dtype=tf.bool)
        g_str_mask = tf.ones_like(pitches, dtype=tf.bool)
        d_str_mask = tf.greater_equal(pitches, 62)
        a_str_mask = tf.greater_equal(pitches, 69)
        e_str_mask = tf.greater_equal(pitches, 76)
        str_mask = tf.stack([null_str_mask, g_str_mask, d_str_mask, a_str_mask, e_str_mask], axis=2)
        return tf.cast(str_mask, tf.float32)

    def create_pos_mask(self, pitch, str_pred):
        # create position mask
        mask = []
        for i in range(pitch.shape[0]):
            for j in range(pitch.shape[1]):
                if pitch[i][j] == 55: # open g string
                    mask.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
                    continue
                elif pitch[i][j] <=58: # g string 1st pos
                    mask.append([1,1,0,0,0,0,0,0,0,0,0,0,0])
                    continue
                elif pitch[i][j] == 59:
                    mask.append([1,1,1,0,0,0,0,0,0,0,0,0,0])
                    continue
                elif pitch[i][j] <= 61:
                    mask.append([1,1,1,1,0,0,0,0,0,0,0,0,0])
                    continue
                elif pitch[i][j] == 62:
                    if str_pred[i][j] ==1:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 63:
                    if str_pred[i][j] ==1:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 64:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,1,1,1,1,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 66:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,1,1,1,1,0,0,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,1,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 68:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,1,1,1,1,0,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,1,1,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 69:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,1,1,1,1,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 70:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,1,1,1,1,0,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 71:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,1,1,1,1,0,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,1,1,1,1,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 73:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,0,1,1,1,1,0,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,0,1,1,1,1,0,0,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,1,1,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 75:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,0,0,1,1,1,1,0])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,0,0,1,1,1,1,0,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,1,1,1,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 76:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,0,0,0,1,1,1,1])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,1,1,1,1,0,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 78:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,0,0,0,0,1,1,1])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,1,1,1,1,0,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,1,1,1,1,0,0,0,0,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 80:
                    if str_pred[i][j] ==1:
                        mask.append([0,0,0,0,0,0,0,0,0,0,0,1,1])
                    elif str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,0,1,1,1,1,0,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,0,1,1,1,1,0,0,0,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,1,1,0,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 82:
                    if str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,0,0,1,1,1,1,0])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,0,0,1,1,1,1,0,0,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,1,1,1,0,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 83:
                    if str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,0,0,0,1,1,1,1])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,1,1,1,1,0,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,1,1,1,1,0,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 85:
                    if str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,0,0,0,0,1,1,1])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,1,1,1,1,0,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,1,1,1,1,0,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 87:
                    if str_pred[i][j] ==2:
                        mask.append([0,0,0,0,0,0,0,0,0,0,0,1,1])
                    elif str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,0,1,1,1,1,0,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,0,1,1,1,1,0,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 88:
                    if str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,0,0,1,1,1,1,0])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,0,0,1,1,1,1,0,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 90:
                    if str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,0,0,0,1,1,1,1])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,1,1,1,1,0,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 92:
                    if str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,0,0,0,0,1,1,1])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,1,1,1,1,0,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 94:
                    if str_pred[i][j] ==3:
                        mask.append([0,0,0,0,0,0,0,0,0,0,0,1,1])
                    elif str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,0,1,1,1,1,0,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 95:
                    if str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,0,0,1,1,1,1,0])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 97:
                    if str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,0,0,0,1,1,1,1])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] <= 99:
                    if str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,0,0,0,0,1,1,1])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                elif pitch[i][j] == 100:
                    if str_pred[i][j] ==4:
                        mask.append([0,0,0,0,0,0,0,0,0,0,0,1,1])
                    else:
                        mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
                    continue
                else:
                    mask.append([1,1,1,1,1,1,1,1,1,1,1,1,1])
        return np.reshape(mask, newshape=[pitch.shape[0], self.n_steps, self.n_pos_classes])


    def predict_finger(self, pitch, string, pos, prev_fin):
        # use string and position to predict finger
        #which string = [pitch on first position, pitch on second position, ...]
        g_str = [55,57,59,60,62,64,65,67,69,71,72,74,76]
        d_str = [62,64,65,67,69,71,72,74,76,77,79,81,83]
        a_str = [69,71,72,74,76,77,79,81,83,84,86,88,89]
        e_str = [76,78,79,81,83,84,86,88,89,91,93,95,96]

        #find pitch that matches to index finger
        if string == 1:
            index_fin = g_str[pos]
        elif string ==2:
            index_fin = d_str[pos]
        elif string ==3:
            index_fin = a_str[pos]
        elif string ==4:
            index_fin = e_str[pos]
        else:
            index_fin = self.pitch_for_invalid_note

        #calculate which finger
        if index_fin != self.pitch_for_invalid_note:
            #distance between the exact pitch and index_finger pitch
            distance = pitch - index_fin

            if distance ==0 and pos ==0:
                finger = 0
            elif distance <= 0:
                finger = 1
            elif distance <=2:
                finger = 2
            elif distance <=4:
                finger = 3
            else:
                finger = 4
        else:
            finger = 0#5

        # if pitch == 57:
        #     print('pos =', pos, 'index finger =', index_fin, 'finger =', finger, 'distance =', distance)

        #cases that may have two finger options to choose
        if pos==1 or pos==0:
            if pitch == 82 and string ==4:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(random.choices(select, prob))
                    finger = array[0]
            elif pitch == 75 and string ==3:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(random.choices(select, prob))
                    finger = array[0]
            elif pitch == 68 and string ==2:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(random.choices(select, prob))
                    finger = array[0]
            elif pitch == 61 and string ==1:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(random.choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(random.choices(select, prob))
                    finger = array[0]
        return finger


    def count_fin(self, prev_fin,prev_pitch,pitch,now_fin, now_str, next_str):
        # calculate which finger & string
        str = now_str
        if prev_fin==1:
            if (pitch - prev_pitch) >=0:
                if (pitch - prev_pitch)<=2:
                    fin = 2
                elif (pitch - prev_pitch)<=4:
                    fin = 3
                elif (pitch - prev_pitch)<=6:
                    fin = 4
                #change string if the future string is different
                elif (pitch - prev_pitch)>=8 and next_str!=now_str and now_str!=4:
                    fin=2
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=10 and next_str!=now_str and now_str!=4:
                    fin=3
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=12 and next_str!=now_str and now_str!=4:
                    fin=4
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                else:
                    fin=4
            else:
                if (prev_pitch - pitch)<=2 and next_str!=now_str and now_str!=1:
                    fin=4
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)<=4 and next_str!=now_str and now_str!=1:
                    fin=3
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)<=6 and next_str!=now_str and now_str!=1:
                    fin=2
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)<=2:
                    fin = 2
                else:
                    fin = 3
        elif prev_fin==2:
            if (pitch - prev_pitch) >=0:
                if (pitch - prev_pitch)<=2:
                    fin = 3
                elif (pitch - prev_pitch)<=4:
                    fin = 4
                elif (pitch - prev_pitch)<=6 and next_str!=now_str and now_str!=4:
                    fin=1
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=8 and next_str!=now_str and now_str!=4:
                    fin=3
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=10 and next_str!=now_str and now_str!=4:
                    fin=4
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                else:
                    fin = 3
            else:
                if (prev_pitch - pitch)<=2:
                    fin = 1
                elif (prev_pitch - pitch)<=4 and next_str!=now_str  and now_str!=1:
                    fin=4
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)<=6 and next_str!=now_str and now_str!=1:
                    fin=3
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)>=8 and next_str!=now_str and now_str!=1:
                    fin=1
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                else:
                    fin = 3
        elif prev_fin ==3:
            if (pitch - prev_pitch) >=0:
                if (pitch - prev_pitch)<=2:
                    fin=4
                elif (pitch - prev_pitch)<=4 and next_str!=now_str and now_str!=4:
                    fin=1
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=6 and next_str!=now_str and now_str!=4:
                    fin=2
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)>=8 and next_str!=now_str and now_str!=4:
                    fin=4
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                else:
                    fin = 4
            else:
                if (prev_pitch - pitch)<=2:
                    fin = 2
                elif (prev_pitch - pitch)<=4:
                    fin = 1
                elif (prev_pitch - pitch)<=6 and next_str!=now_str and now_str!=1:
                    fin=4
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)>=8 and next_str!=now_str and now_str!=1:
                    fin=2
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)>=10 and next_str!=now_str and now_str!=1:
                    fin=1
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                else:
                    fin = 4
        elif prev_fin==4:
            if (pitch - prev_pitch) >=0:
                if (pitch - prev_pitch)<=2 and next_str!=now_str and now_str!=4:
                    fin=1
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)<=4 and next_str!=now_str and now_str!=4:
                    fin=2
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                elif (pitch - prev_pitch)<=6 and next_str!=now_str and now_str!=4:
                    fin=3
                    if now_str ==1:
                        str = 2
                    elif now_str ==2:
                        str = 3
                    else:
                        str = 4
                else:
                    fin=3
            else:
                if (prev_pitch - pitch)<=2:
                    fin = 3
                elif (prev_pitch - pitch)<=4:
                    fin = 2
                elif (prev_pitch - pitch)<=6:
                    fin = 1
                elif (prev_pitch - pitch)>=8 and next_str!=now_str and now_str!=1:
                    fin=3
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)>=10 and next_str!=now_str and now_str!=1:
                    fin=2
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                elif (prev_pitch - pitch)>=12 and next_str!=now_str and now_str!=1:
                    fin=1
                    if now_str ==4:
                        str = 3
                    elif now_str ==3:
                        str = 2
                    else:
                        str = 1
                else:
                    fin=1
        else: # prev_fin==0:
            fin = now_fin

        return fin, str


    def avoid_same_finger(self, str, pos, fin, pitch):
        # change finger if there are >=3 same finger
        new_fin = []
        new_str = []
        prev_fin = 0
        prev_pitch = pitch[0][0]
        prev_pos = 0
        flag = 0

        for i in range(pitch.shape[0]):
            for j in range(self.n_steps):
                #count repeat finger
                if prev_fin == fin[i][j]:
                    flag=flag+1
                    #if same pitch then don't count
                    if prev_pitch==pitch[i][j]:
                        flag = flag-1
                    #repeat three time
                    if flag==2:
                        flag = 0
                        if j==31 and i!=pitch.shape[0]-1:
                            next_str = str[i+1][0]
                        elif j!=31:
                            next_str = str[i][j+1]
                        else:
                            next_str = str[i][j]
                        fin[i][j], str[i][j] = self.count_fin(prev_fin,prev_pitch,pitch[i][j], fin[i][j], str[i][j],next_str)
                else:
                    flag = 0

                new_fin.append(fin[i][j])
                new_str.append(str[i][j])
                prev_fin = fin[i][j]
                prev_pitch = pitch[i][j]

        new_str = np.reshape(new_str, newshape=pitch.shape)
        new_fin = np.reshape(new_fin, newshape=pitch.shape)
        return new_str, pos, new_fin

    def decode_pos(self, pos_scores, mode='basic', k=3):
        # get the nearest position w.r.t. the previous position from the top-k positions
        top_k_pos = (np.argsort(pos_scores, axis=2)[:, :, ::-1])[:, :, :k] # [batch, n_steps, 3]

        if mode == 'basic':
            out_pos = np.argmax(pos_scores, axis=2)
        elif mode == 'nearest':
            out_pos = np.zeros_like(pos_scores[:, :, 0], dtype=np.int32) # [batch, n_steps]
            out_pos[:, 0] = np.argmax(pos_scores[:, 0, :], axis=1) # [batch]
            for j in range(1, pos_scores.shape[1]):
                prev_pos = out_pos[:, j-1] # [batch]
                dist = top_k_pos[:, j, :] - prev_pos[:,None] # [batch, 3]
                minarg = np.argmin(np.abs(dist), axis=1) # [batch]
                minarg = (np.arange(len(minarg)), minarg)
                out_pos[:, j] = top_k_pos[:, j, :][minarg] # [batch]
        elif mode == 'lowest':
            # out_pos = np.min(top_k_pos, axis=2)
            best_pos = np.argmax(pos_scores, axis=2)
            if_best_in_top_k = np.any(top_k_pos == best_pos[:,:,None], axis=2)
            out_pos = np.min(top_k_pos, axis=2)
            out_pos[if_best_in_top_k] = best_pos[if_best_in_top_k]
        else:
            print('Error: invalid mode.')
            exit(1)
        return out_pos

    def train(self):
        # Load data
        corpus = self.load_data()
        corpus = self.segment_corpus(corpus)  # {key: {segments: 2d_array, lens: len_list}}
        # Get cross-validation sets
        X, Y = self.create_training_and_testing_sets(corpus)
        n_train_samples = X['train']['pitch'].shape[0]
        n_test_samples = X['test']['pitch'].shape[0]
        n_iterations_per_epoch = int(np.ceil(n_train_samples / self.batch_size))
        print('n_iterations_per_epoch=', n_iterations_per_epoch)

        # Placeholders
        x_p = tf.placeholder(tf.int32, [None, self.n_steps], name="pitch")
        x_s = tf.placeholder(tf.float32, [None, self.n_steps], name="start")
        x_d = tf.placeholder(tf.float32, [None, self.n_steps], name="duration")
        x_b = tf.placeholder(tf.int32, [None, self.n_steps], name="beat_type")
        x_len = tf.placeholder(tf.int32, [None], name="valid_lens")
        y_s = tf.placeholder(tf.int32, [None, self.n_steps], name="string")
        y_p = tf.placeholder(tf.int32, [None, self.n_steps], name="position")
        y_f = tf.placeholder(tf.int32, [None, self.n_steps], name="finger")
        f_s = tf.placeholder(tf.float32, name="f_score_string")
        f_p = tf.placeholder(tf.float32, name="f_score_position")
        f_f = tf.placeholder(tf.float32, name="f_score_finger")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        with tf.variable_scope('model'):
            logits_s, logits_p = self.BLSTM(x_p, x_s, x_d, x_b, x_len, dropout)

        with tf.name_scope('loss'):
            seq_mask = tf.sequence_mask(lengths=x_len, maxlen=self.n_steps, dtype=tf.float32)  # [batch, n_steps]
            n_valid = tf.reduce_sum(seq_mask)
            loss_s = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_s, self.n_str_classes), logits=logits_s, weights=seq_mask)
            loss_p = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_p, self.n_pos_classes), logits=logits_p, weights=seq_mask)
            # Total loss
            loss = loss_s + loss_p
        summary_loss = tf.Variable([0.0 for _ in range(3)], trainable=False, dtype=tf.float32)
        summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
        update_loss = tf.assign(summary_loss, summary_loss + n_valid * [loss, loss_s, loss_p])
        update_valid = tf.assign(summary_valid, summary_valid + n_valid)
        mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
        clr_summary_loss = summary_loss.initializer
        clr_summary_valid = summary_valid.initializer
        tf.summary.scalar('Loss_total', summary_loss[0])
        tf.summary.scalar('Loss_string', summary_loss[1])
        tf.summary.scalar('Loss_position', summary_loss[2])

        with tf.name_scope('Optimization'):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(learning_rate=self.initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=n_iterations_per_epoch,
                                                       decay_rate=0.96,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-9)

            train_op = optimizer.minimize(loss)

        with tf.name_scope('Evaluation'):
            str_mask = self.create_str_mask(x_p)
            logits_s = tf.where(tf.equal(str_mask, 1), logits_s, tf.ones_like(logits_s) * - self.INFIN) # masking
            pred_s = tf.argmax(logits_s, axis=2, output_type=tf.int32)
        tf.summary.scalar('F1_s', f_s)
        tf.summary.scalar('F1_p', f_p)
        tf.summary.scalar('F1_f', f_f)

        print('Saving model to: %s' % self.save_dir)
        train_writer = tf.summary.FileWriter(self.save_dir + '/train')
        test_writer = tf.summary.FileWriter(self.save_dir + '/test')
        merged = tf.summary.merge_all()
        train_writer.add_graph(tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=1)

        # Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            startTime = time.time() # start time of training
            best_test_score = [0.0 for _ in range(3)]
            in_succession = 0
            best_epoch = 0
            # Batched indices
            indices = np.arange(n_train_samples)
            batch_indices = [indices[x:x + self.batch_size] for x in range(0, len(indices), self.batch_size)]
            step = 0
            for epoch in range(self.n_epochs):
                if epoch > 0:
                    # Shuffle training data
                    indices = np.array(random.sample(range(n_train_samples), n_train_samples))
                    batch_indices = np.array([indices[x:x + self.batch_size] for x in range(0, len(indices), self.batch_size)])

                # Batched training
                train_str, train_pos_scores = [], []
                for idx in batch_indices:
                    batch = (X['train']['pitch'][idx],
                             X['train']['start'][idx],
                             X['train']['duration'][idx],
                             X['train']['beat_type'][idx],
                             X['train']['lens'][idx],
                             Y['train']['string'][idx],
                             Y['train']['position'][idx],
                             Y['train']['finger'][idx])

                    train_run_list = [train_op, update_valid, update_loss, loss, loss_s, loss_p, pred_s, logits_p]
                    train_feed_dict = {x_p: batch[0],
                                       x_s: batch[1],
                                       x_d: batch[2],
                                       x_b: batch[3],
                                       x_len: batch[4],
                                       y_s: batch[5],
                                       y_p: batch[6],
                                       y_f: batch[7],
                                       dropout: self.drop,
                                       global_step: step + 1}
                    _, _, _, train_loss, train_loss_s, train_loss_p, train_pred_s, train_logits_p = sess.run(train_run_list, feed_dict=train_feed_dict)
                    train_str.append(train_pred_s)
                    train_pos_scores.append(train_logits_p)
                    if step == 0:
                        print('*~ loss_s %.4f, loss_p %.4f, ~*' % (train_loss_s, train_loss_p))
                    step += 1
                train_str = np.concatenate(train_str, axis=0)
                train_pos_scores = np.concatenate(train_pos_scores, axis=0)
                # Recovery ordering
                gather_id = [np.where(indices == ord)[0][0] for ord in range(n_train_samples)]
                train_str = train_str[gather_id, :]
                train_pos_scores = train_pos_scores[gather_id, :]
                # Get position mask
                train_pos_mask = self.create_pos_mask(X['train']['pitch'], train_str)
                # Get masked positions
                train_pos_scores[train_pos_mask == 0] = -self.INFIN
                train_pos = np.argmax(train_pos_scores, axis=2)
                # Decode fingers
                train_fin = []
                prev_train_fin = 0
                for i in range(n_train_samples):
                    for j in range(self.n_steps):
                        prev_train_fin = self.predict_finger(X['train']['pitch'][i][j], train_str[i][j], train_pos[i][j], prev_train_fin)
                        train_fin.append(prev_train_fin)
                train_fin = np.reshape(train_fin, newshape=[n_train_samples, self.n_steps])
                # Change str and fin to avoid using the same finger
                train_str, train_pos, train_fin = self.avoid_same_finger(train_str, train_pos, train_fin, X['train']['pitch'])
                # Calculate performance
                boolean_mask = np.arange(self.n_steps)[None, :] < X['train']['lens'][:, None]
                train_P_s, train_R_s, train_F_s, _ = precision_recall_fscore_support(y_true=Y['train']['string'][boolean_mask], y_pred=train_str[boolean_mask], average='micro')
                train_P_p, train_R_p, train_F_p, _ = precision_recall_fscore_support(y_true=Y['train']['position'][boolean_mask], y_pred=train_pos[boolean_mask], average='micro')
                train_P_f, train_R_f, train_F_f, _ = precision_recall_fscore_support(y_true=Y['train']['finger'][boolean_mask], y_pred=train_fin[boolean_mask], average='micro')

                # Display training log
                _, train_losses, train_summary = sess.run([mean_loss, summary_loss, merged], feed_dict={f_s: train_F_s, f_p: train_F_p, f_f: train_F_f})
                sess.run([clr_summary_valid, clr_summary_loss]) # clear summaries
                train_writer.add_summary(train_summary, epoch)
                print("==== epoch %d: train_loss %.4f (s %.4f, p %.4f), train_F1: s %.4f, p %.4f, f %.4f ====" % (epoch, train_losses[0], train_losses[1], train_losses[2], train_F_s, train_F_p, train_F_f))
                sample_id = random.randint(0, n_train_samples - 1)
                print('len'.ljust(6, ' '), X['train']['lens'][sample_id])
                print('x_p'.ljust(4, ' '), ''.join([pretty_midi.note_number_to_name(p).rjust(4, ' ') if p < self.pitch_for_invalid_note else 'X'.rjust(4, ' ') for p in X['train']['pitch'][sample_id]]))
                print('y_s'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in Y['train']['string'][sample_id]]))
                print('o_s'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in train_str[sample_id]]))
                print('y_p'.ljust(4, ' '), ''.join([str(p).rjust(4, ' ') for p in Y['train']['position'][sample_id]]))
                print('o_p'.ljust(4, ' '), ''.join([str(p).rjust(4, ' ') for p in train_pos[sample_id]]))
                print('y_f'.ljust(4, ' '), ''.join([str(f).rjust(4, ' ') for f in Y['train']['finger'][sample_id]]))
                print('o_f'.ljust(4, ' '), ''.join([str(f).rjust(4, ' ') for f in train_fin[sample_id]]))

                # Testing
                test_run_list = [update_valid, update_loss, pred_s, logits_p]
                test_feed_fict = {x_p: X['test']['pitch'],
                                  x_s: X['test']['start'],
                                  x_d: X['test']['duration'],
                                  x_b: X['test']['beat_type'],
                                  x_len: X['test']['lens'],
                                  y_s: Y['test']['string'],
                                  y_p: Y['test']['position'],
                                  y_f: Y['test']['finger'],
                                  dropout: 0}
                _, _, test_str, test_pos_scores = sess.run(test_run_list, feed_dict=test_feed_fict)

                # Get position mask
                test_pos_mask = self.create_pos_mask(X['test']['pitch'], test_str)
                # Decode positions using different modes
                test_pos_scores[test_pos_mask == 0] = -self.INFIN
                test_pos_basic = np.argmax(test_pos_scores, axis=2)
                test_pos_lowest = self.decode_pos(test_pos_scores, mode='lowest')
                test_pos_nearest = self.decode_pos(test_pos_scores, mode='nearest')

                # Decode fingers
                test_fin_basic, test_fin_lowest, test_fin_nearest = [], [], []
                prev_test_fin_basic = prev_test_fin_lowest = prev_test_fin_nearest = 0
                for i in range(n_test_samples):
                    for j in range(self.n_steps):
                        prev_test_fin_basic = self.predict_finger(X['test']['pitch'][i][j], test_str[i][j], test_pos_basic[i][j], prev_test_fin_basic)
                        prev_test_fin_lowest = self.predict_finger(X['test']['pitch'][i][j], test_str[i][j], test_pos_lowest[i][j], prev_test_fin_lowest)
                        prev_test_fin_nearest = self.predict_finger(X['test']['pitch'][i][j], test_str[i][j], test_pos_nearest[i][j], prev_test_fin_nearest)
                        test_fin_basic.append(prev_test_fin_basic)
                        test_fin_lowest.append(prev_test_fin_lowest)
                        test_fin_nearest.append(prev_test_fin_nearest)
                test_fin_basic = np.reshape(test_fin_basic, newshape=[n_test_samples, self.n_steps])
                test_fin_lowest = np.reshape(test_fin_lowest, newshape=[n_test_samples, self.n_steps])
                test_fin_nearest = np.reshape(test_fin_nearest, newshape=[n_test_samples, self.n_steps])
                # Change str and fin to avoid using the same finger
                test_str_basic, test_pos_basic, test_fin_basic = self.avoid_same_finger(test_str, test_pos_basic, test_fin_basic, X['test']['pitch'])
                test_str_lowest, test_pos_lowest, test_fin_lowest = self.avoid_same_finger(test_str, test_pos_lowest, test_fin_lowest, X['test']['pitch'])
                test_str_nearest, test_pos_nearest, test_fin_nearest = self.avoid_same_finger(test_str, test_pos_nearest, test_fin_nearest, X['test']['pitch'])

                # Calculate performance
                boolean_mask = np.arange(self.n_steps)[None, :] < X['test']['lens'][:, None]
                test_P_s, test_R_s, test_F_s, _ = precision_recall_fscore_support(y_true=Y['test']['string'][boolean_mask], y_pred=test_str_basic[boolean_mask], average='micro')
                test_P_p, test_R_p, test_F_p, _ = precision_recall_fscore_support(y_true=Y['test']['position'][boolean_mask], y_pred=test_pos_basic[boolean_mask], average='micro')
                test_P_f, test_R_f, test_F_f, _ = precision_recall_fscore_support(y_true=Y['test']['finger'][boolean_mask], y_pred=test_fin_basic[boolean_mask], average='micro')

                _, test_losses, test_summary = sess.run([mean_loss, summary_loss, merged], feed_dict={f_s: test_F_s, f_p: test_F_p, f_f: test_F_f})
                sess.run([clr_summary_valid, clr_summary_loss]) # clear summaries
                test_writer.add_summary(test_summary, epoch)
                print("----  epoch %d: test_loss %.4f (s %.4f, p %.4f), test_F1: s %.4f, p %.4f, f %.4f ----" % (epoch, test_losses[0], test_losses[1], test_losses[2], test_F_s, test_F_p, test_F_f))
                sample_id = random.randint(0, n_test_samples - 1)
                print('len'.ljust(6, ' '), X['test']['lens'][sample_id])
                print('x_p'.ljust(4, ' '), ''.join([pretty_midi.note_number_to_name(p).rjust(4,  ' ') if p < self.pitch_for_invalid_note else 'X'.rjust(4, ' ') for p in X['test']['pitch'][sample_id]]))
                print('y_s'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in Y['test']['string'][sample_id]]))
                print('o_s'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in test_str[sample_id]]))
                print('y_p'.ljust(4, ' '), ''.join([str(p).rjust(4, ' ') for p in Y['test']['position'][sample_id]]))
                print('o_p'.ljust(4, ' '), ''.join([str(p).rjust(4, ' ') for p in test_pos_basic[sample_id]]))
                print('y_f'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in Y['test']['finger'][sample_id]]))
                print('o_f'.ljust(4, ' '), ''.join([str(s).rjust(4, ' ') for s in test_fin_basic[sample_id]]))

                # Check if early stopping
                if (test_F_s + test_F_p) > sum(best_test_score[:2]):
                    best_test_score = [test_F_s, test_F_p, test_F_f]
                    best_epoch = epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...')
                    saver.save(sess, self.save_dir + '\\violin_fingering_estimation.ckpt')
                else:
                    in_succession += 1
                    if in_succession > self.n_in_succession:
                        print('Early stopping.')
                        break
            elapsed_time = time.time() - startTime
            np.set_printoptions(precision=4)
            print('training time = %.2f hr' % (elapsed_time / 3600))
            print('best epoch = ', best_epoch)
            print('best test score =', np.round(best_test_score, 4))

    def inference(self, pitches, starts, durations, beat_types, strings, positions, fingers, mode='basic'):

        if mode not in ['basic', 'lowest', 'nearest']:
            print('invalid mode.')
            exit(1)

        tf.reset_default_graph()

        n_infer_samples = 1
        lens = np.array([len(pitches)])

        n_pad = self.n_steps - len(pitches)
        pitches = np.pad(pitches, pad_width=[(0, n_pad)], mode='constant')[None,:]
        starts = np.pad(starts, pad_width=[(0, n_pad)], mode='constant')[None,:]
        durations = np.pad(durations, pad_width=[(0, n_pad)], mode='constant')[None,:]
        beat_types = np.pad(beat_types, pad_width=[(0, n_pad)], mode='constant')[None,:]
        strings = np.pad(strings, pad_width=[(0, n_pad)], mode='constant')[None, :]
        positions = np.pad(positions, pad_width=[(0, n_pad)], mode='constant')[None, :]

        # Placeholders
        # dt = [('pitch', int), ('start', float), ('duration', float), ('beat_type', int), ('string', int), ('position', int), ('finger', int)]
        x_p = tf.placeholder(tf.int32, [None, self.n_steps], name="pitch")
        x_s = tf.placeholder(tf.float32, [None, self.n_steps], name="start")
        x_d = tf.placeholder(tf.float32, [None, self.n_steps], name="duration")
        x_b = tf.placeholder(tf.int32, [None, self.n_steps], name="beat_type")
        x_len = tf.placeholder(tf.int32, [None], name="valid_lens")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")

        with tf.variable_scope('model'):
            logits_s, logits_p = self.BLSTM(x_p, x_s, x_d, x_b, x_len, dropout)

            # Strings
            str_mask = self.create_str_mask(x_p)
            logits_s = tf.where(tf.equal(str_mask, 1), logits_s, tf.ones_like(logits_s) * -self.INFIN) # masking
            pred_s = tf.argmax(logits_s, axis=2, output_type=tf.int32) # predicted strings

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_dir)

            feed_fict = {x_p: pitches,
                         x_s: starts,
                         x_d: durations,
                         x_b: beat_types,
                         x_len: lens,
                         dropout: 0}
            str_pred, pos_scores = sess.run([pred_s, logits_p], feed_dict=feed_fict)

        # Use user specified strings
        user_str_mask = (strings != 0)
        str_pred[user_str_mask] = strings[user_str_mask]

        # Positions
        pos_mask = self.create_pos_mask(pitches, str_pred)
        pos_scores[pos_mask == 0] = -self.INFIN
        # Adjust scores according to user specified positions
        user_pos_mask = np.eye(self.n_pos_classes, dtype=int)[positions]
        user_pos_mask[:,:,0] = 0
        pos_scores[user_pos_mask==1] = self.INFIN
        pos_basic = self.decode_pos(pos_scores, mode='basic')
        pos_lowest = self.decode_pos(pos_scores, mode='lowest')
        pos_nearest = self.decode_pos(pos_scores, mode='nearest')

        # Fingers
        fin_basic, fin_lowest, fin_nearest = [], [], []
        prev_fin_basic = prev_fin_lowest = prev_fin_nearest = 0
        for i in range(n_infer_samples):
            for j in range(self.n_steps):
                prev_fin_basic = self.predict_finger(pitches[i][j], str_pred[i][j], pos_basic[i][j], prev_fin_basic)
                prev_fin_lowest = self.predict_finger(pitches[i][j], str_pred[i][j], pos_lowest[i][j], prev_fin_lowest)
                prev_fin_nearest = self.predict_finger(pitches[i][j], str_pred[i][j], pos_nearest[i][j], prev_fin_nearest)
                fin_basic.append(prev_fin_basic)
                fin_lowest.append(prev_fin_lowest)
                fin_nearest.append(prev_fin_nearest)
        fin_basic = np.reshape(fin_basic, newshape=[n_infer_samples, self.n_steps])
        fin_lowest = np.reshape(fin_lowest, newshape=[n_infer_samples, self.n_steps])
        fin_nearest = np.reshape(fin_nearest, newshape=[n_infer_samples, self.n_steps])

        # Change str and fin to avoid using the same finger
        str_basic, pos_basic, fin_basic = self.avoid_same_finger(str_pred, pos_basic, fin_basic, pitches)
        str_lowest, pos_lowest, fin_lowest = self.avoid_same_finger(str_pred, pos_lowest, fin_lowest, pitches)
        str_nearest, pos_nearest, fin_nearest = self.avoid_same_finger(str_pred, pos_nearest, fin_nearest, pitches)

        if mode == 'basic':
            return str_basic, pos_basic, fin_basic
        elif mode == 'lowest':
            return str_lowest, pos_lowest, fin_lowest
        else: # mode == 'nearest'
            return str_nearest, pos_nearest, fin_nearest

if __name__ == '__main__':

    # Training
    model = model = violin_fingering_model()
    model.train()

    # # Inference
    # pitches = [55, 57, 59, 60, 62, 64, 66, 67] # G scale
    # n_events = len(pitches)
    # starts = [i * 1 for i in range(n_events)]
    # durations = [1 for _ in range(n_events)]
    # beat_types = [3 for _ in range(n_events)] # {'': 0, '1th': 1, '2th': 2, '4th': 3, '8th': 4, '16th': 5, '32th': 6}
    # strings = [0 for _ in range(n_events)]
    # positions = [0 for _ in range(n_events)]
    # fingers = [0 for _ in range(n_events)]
    #
    # model = violin_fingering_model()
    # pred_str, pred_pos, pred_fin = model.inference(pitches=pitches,
    #                                                starts=starts,
    #                                                durations=durations,
    #                                                beat_types=beat_types,
    #                                                strings=strings,
    #                                                positions=positions,
    #                                                fingers=fingers,
    #                                                mode='basic') # valid mode = {'basic', 'lowest', 'nearest'}
    #
    # # Print the estimations
    # string_classes = ['N', 'G', 'D', 'A', 'E']
    # n_notes = len(pitches)
    # print('pitch'.ljust(9), ''.join([pretty_midi.note_number_to_name(number).rjust(4) for number in pitches]))
    # print('string'.ljust(9), ''.join([string_classes[s].rjust(4) for s in pred_str[0, :n_notes]]))
    # print('position'.ljust(9), ''.join([str(p).rjust(4) for p in pred_pos[0, :n_notes]]))
    # print('finger'.ljust(9), ''.join([str(f).rjust(4) for f in pred_fin[0, :n_notes]]))