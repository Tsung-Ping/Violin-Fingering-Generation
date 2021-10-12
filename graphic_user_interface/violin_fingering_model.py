import os
import numpy as np
from random import choices
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMBlockCell
from model_downloader import download_pretrained_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class violin_fingering_model(object):
    def __init__(self):
        self.pitch_for_invalid_note = 101
        self.lowest_pitch = 55
        self.n_p_classes = 46 + 1 # pitch range =  55 to 100, pitch_for_invalid_note = 101
        self.n_b_classes = 7 # {'', '1th', '2th', '4th',  '8th',  '16th', '32th'}
        self.n_str_classes = 5 # {'', G', 'D', 'A', 'E'}
        self.n_pos_classes = 13 # {'', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}
        self.n_fin_classes = 5 # {'' & '0', '1', '2', '3', '4'}
        self.n_steps = 32
        self.embedding_size = 100
        self.hidden_size = 100
        self.INFIN = 1e12 # infinite number
        self.model_dir = './TNUA_violin.ckpt'

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
                        array = np.array(choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(choices(select, prob))
                    finger = array[0]
            elif pitch == 75 and string ==3:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(choices(select, prob))
                    finger = array[0]
            elif pitch == 68 and string ==2:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(choices(select, prob))
                    finger = array[0]
            elif pitch == 61 and string ==1:
                if prev_fin == finger:
                    if finger ==3:
                        select = [3,4]
                        prob = [0.2,0.8]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                    else:
                        select = [3,4]
                        prob = [0.8,0.2]
                        array = np.array(choices(select, prob))
                        finger = array[0]
                else:
                    select = [3,4]
                    prob = [0.8,0.2]
                    array = np.array(choices(select, prob))
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

    def inference(self, pitches, starts, durations, beat_types, strings, positions, fingers, mode='basic'):

        if mode not in ['basic', 'lowest', 'nearest']:
            print('invalid mode.')
            exit(1)

        tf.reset_default_graph()

        n_infer_samples = 1
        lens = np.array([len(pitches)])

        self.n_steps = len(pitches)
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




