
"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils_cmu

class Seq2SeqModel(object):
  """Sequence-to-sequence model for human motion prediction"""

  def __init__(self,
               source_seq_len,
               target_seq_len,
               rnn_size, # hidden recurrent layer size
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               number_of_actions,
               one_hot=False,
               reconstruction_len=5,
               anchor_len=5,
               sub_loss_weight=0.1,
               loss_weight_decay_factor=1.0,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      reconstruction_len: length of subsequence to retrospect
      anchor_len: interval size of anchor points
      sub_loss_weight: alpha to balance retrospection loss and main loss
      loss_weight_decay_factor: decay learning rate by this much when needed.
      dtype: the data type to use to store internal variables.
    """

    self.HUMAN_SIZE = 70
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'train')))
    self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'test')))

    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.sub_loss_weight = tf.Variable( float(sub_loss_weight), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.sub_loss_weight_decay_op = self.sub_loss_weight.assign( self.sub_loss_weight * loss_weight_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)

    # === Create the RNN that will keep the state ===
    print('rnn_size = {0}'.format( rnn_size ))
    cell = tf.contrib.rnn.GRUCell( self.rnn_size )

    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(num_layers)] )

    # === Transform the inputs ===
    with tf.name_scope("inputs"):

      enc_in = tf.placeholder(dtype, shape=[None, source_seq_len-1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_out")
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out
      self.keep_prob = keep_prob

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])

      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      dec_in = tf.reshape(dec_in, [-1, self.input_size])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])

      enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
      dec_in = tf.split(dec_in, target_seq_len, axis=0)
      dec_out = tf.split(dec_out, target_seq_len, axis=0)

    # Store the outputs here
    outputs  = []

    # Init weights and bias for spatial attention on main RNN
    with tf.variable_scope("spatial_attention", reuse=tf.AUTO_REUSE):
      # Init the weight for spatial attention (input, hidden state and score)
      SA_I_w = tf.get_variable("SA_I_w", [self.input_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      SA_I_b = tf.get_variable("SA_I_b", [self.input_size], dtype=tf.float32,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      SA_H_w = tf.get_variable("SA_H_w", [self.rnn_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      SA_S_w = tf.get_variable("SA_S_w", [self.input_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      SA_S_b = tf.get_variable("SA_S_b", [self.input_size], dtype=tf.float32,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    # Build the RNN
    with vs.variable_scope("basic_rnn_seq2seq"):
      # _, enc_state = tf.contrib.rnn.static_rnn(cell, enc_in, dtype=tf.float32) # Encoder
      enc_states = custom_encoder_net(cell, enc_in, num_layers, self.keep_prob, self.input_size, dtype=tf.float32, scope='custom_encoder_net') # Encoder
      # outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder( dec_in, enc_state, cell, loop_function=lf ) # Decoder
      outputs, dec_states = custom_decoder_net(dec_in, enc_states, cell, source_seq_len-1, target_seq_len, num_layers, self.keep_prob, self.input_size, -1, scope='custom_decoder_net') #Decoder


    self.outputs = outputs

    with tf.name_scope("loss_angles"):
      loss_angles = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))

    # Init sub_loss and construct whole_sequence for next step
    sub_loss = 0.0
    whole_sequence = tf.concat([enc_in,dec_in,tf.expand_dims(dec_out[-1],0)], 0)
    # self.states = tf.concat([enc_states, dec_states], 0)
    self.states = []
    self.states.extend(enc_states)
    self.states.extend(dec_states)

    # Select anchor points for whole sequence
    with tf.name_scope("anchor_points"):
      anchor_points = []
      for i in range(reconstruction_len-1, target_seq_len+source_seq_len-1):
      # for i in range(reconstruction_len-1, source_seq_len-1):
      # for i in range(source_seq_len-1, target_seq_len+source_seq_len-1):
        if i % anchor_len == anchor_len-1:
          anchor_points.append([self.states[i],i])

    # Init weights and bias for spatial attention on subsequence

    # for idx in range(len(anchor_points)):
    #   with tf.variable_scope("sub_spatial_attention"+str(idx), reuse=tf.AUTO_REUSE):
    with tf.variable_scope("sub_spatial_attention", reuse=tf.AUTO_REUSE):
      SA_I_w = tf.get_variable("SA_I_w", [self.input_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      SA_I_b = tf.get_variable("SA_I_b", [self.input_size], dtype=tf.float32,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      SA_H_w = tf.get_variable("SA_H_w", [self.rnn_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      SA_S_w = tf.get_variable("SA_S_w", [self.input_size,self.input_size], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      SA_S_b = tf.get_variable("SA_S_b", [self.input_size], dtype=tf.float32,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    # Construct sub-sequence for each anchor point
    with vs.variable_scope("basic_rnn_seq2seq"):
      for idx in range(len(anchor_points)):
        LSTMstate = anchor_points[idx][0]
        index = anchor_points[idx][1]
        sub_enc_states = []
        sub_enc_states.append(LSTMstate)

        # Sub-sequence outputs
        sub_dec_in = whole_sequence[index-reconstruction_len+1:index]
        subOutputs, _ = custom_decoder_net(sub_dec_in, sub_enc_states, cell, source_seq_len-1,
         reconstruction_len - 1, num_layers, self.keep_prob, self.input_size, idx, scope='sub_custom_decoder_net') # Decoder
        
        # Compute loss for sub-sequence
        sub_dec_out = whole_sequence[index-reconstruction_len+2:index+1]
        sub_loss_angles = tf.reduce_mean(tf.square(tf.subtract(sub_dec_out, subOutputs)))

        with tf.name_scope("sub_loss_angles"):
          sub_loss += sub_loss_angles

    # Compute mean sub_loss
    self.main_loss = loss_angles
    # self.velocities_loss = loss_velocities

    # Return training loss or validation loss
    self.training_loss = loss_angles + self.sub_loss_weight * sub_loss
    self.validation_loss = loss_angles


    self.loss_summary = tf.summary.scalar('loss/loss', self.validation_loss)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)

    # Update all the trainable parameters
    gradients = tf.gradients(self.training_loss, params)

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # === variables for loss in Euler Angles -- for each action
    with tf.name_scope( "euler_error_walking" ):
      self.walking_err80   = tf.placeholder( tf.float32, name="walking_srnn_seeds_0080" )
      self.walking_err160  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0160" )
      self.walking_err320  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0320" )
      self.walking_err400  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0400" )
      self.walking_err560  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0560" )
      self.walking_err1000 = tf.placeholder( tf.float32, name="walking_srnn_seeds_1000" )

      self.walking_err80_summary   = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0080', self.walking_err80 )
      self.walking_err160_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0160', self.walking_err160 )
      self.walking_err320_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0320', self.walking_err320 )
      self.walking_err400_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0400', self.walking_err400 )
      self.walking_err560_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0560', self.walking_err560 )
      self.walking_err1000_summary = tf.summary.scalar( 'euler_error_walking/srnn_seeds_1000', self.walking_err1000 )
    with tf.name_scope( "euler_error_eating" ):
      self.eating_err80   = tf.placeholder( tf.float32, name="eating_srnn_seeds_0080" )
      self.eating_err160  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0160" )
      self.eating_err320  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0320" )
      self.eating_err400  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0400" )
      self.eating_err560  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0560" )
      self.eating_err1000 = tf.placeholder( tf.float32, name="eating_srnn_seeds_1000" )

      self.eating_err80_summary   = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0080', self.eating_err80 )
      self.eating_err160_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0160', self.eating_err160 )
      self.eating_err320_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0320', self.eating_err320 )
      self.eating_err400_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0400', self.eating_err400 )
      self.eating_err560_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0560', self.eating_err560 )
      self.eating_err1000_summary = tf.summary.scalar( 'euler_error_eating/srnn_seeds_1000', self.eating_err1000 )
    with tf.name_scope( "euler_error_smoking" ):
      self.smoking_err80   = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0080" )
      self.smoking_err160  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0160" )
      self.smoking_err320  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0320" )
      self.smoking_err400  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0400" )
      self.smoking_err560  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0560" )
      self.smoking_err1000 = tf.placeholder( tf.float32, name="smoking_srnn_seeds_1000" )

      self.smoking_err80_summary   = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0080', self.smoking_err80 )
      self.smoking_err160_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0160', self.smoking_err160 )
      self.smoking_err320_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0320', self.smoking_err320 )
      self.smoking_err400_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0400', self.smoking_err400 )
      self.smoking_err560_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0560', self.smoking_err560 )
      self.smoking_err1000_summary = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_1000', self.smoking_err1000 )
    with tf.name_scope( "euler_error_discussion" ):
      self.discussion_err80   = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0080" )
      self.discussion_err160  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0160" )
      self.discussion_err320  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0320" )
      self.discussion_err400  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0400" )
      self.discussion_err560  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0560" )
      self.discussion_err1000 = tf.placeholder( tf.float32, name="discussion_srnn_seeds_1000" )

      self.discussion_err80_summary   = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0080', self.discussion_err80 )
      self.discussion_err160_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0160', self.discussion_err160 )
      self.discussion_err320_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0320', self.discussion_err320 )
      self.discussion_err400_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0400', self.discussion_err400 )
      self.discussion_err560_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0560', self.discussion_err560 )
      self.discussion_err1000_summary = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_1000', self.discussion_err1000 )
    with tf.name_scope( "euler_error_directions" ):
      self.directions_err80   = tf.placeholder( tf.float32, name="directions_srnn_seeds_0080" )
      self.directions_err160  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0160" )
      self.directions_err320  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0320" )
      self.directions_err400  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0400" )
      self.directions_err560  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0560" )
      self.directions_err1000 = tf.placeholder( tf.float32, name="directions_srnn_seeds_1000" )

      self.directions_err80_summary   = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0080', self.directions_err80 )
      self.directions_err160_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0160', self.directions_err160 )
      self.directions_err320_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0320', self.directions_err320 )
      self.directions_err400_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0400', self.directions_err400 )
      self.directions_err560_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0560', self.directions_err560 )
      self.directions_err1000_summary = tf.summary.scalar( 'euler_error_directions/srnn_seeds_1000', self.directions_err1000 )
    with tf.name_scope( "euler_error_greeting" ):
      self.greeting_err80   = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0080" )
      self.greeting_err160  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0160" )
      self.greeting_err320  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0320" )
      self.greeting_err400  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0400" )
      self.greeting_err560  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0560" )
      self.greeting_err1000 = tf.placeholder( tf.float32, name="greeting_srnn_seeds_1000" )

      self.greeting_err80_summary   = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0080', self.greeting_err80 )
      self.greeting_err160_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0160', self.greeting_err160 )
      self.greeting_err320_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0320', self.greeting_err320 )
      self.greeting_err400_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0400', self.greeting_err400 )
      self.greeting_err560_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0560', self.greeting_err560 )
      self.greeting_err1000_summary = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_1000', self.greeting_err1000 )
    with tf.name_scope( "euler_error_phoning" ):
      self.phoning_err80   = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0080" )
      self.phoning_err160  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0160" )
      self.phoning_err320  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0320" )
      self.phoning_err400  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0400" )
      self.phoning_err560  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0560" )
      self.phoning_err1000 = tf.placeholder( tf.float32, name="phoning_srnn_seeds_1000" )

      self.phoning_err80_summary   = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0080', self.phoning_err80 )
      self.phoning_err160_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0160', self.phoning_err160 )
      self.phoning_err320_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0320', self.phoning_err320 )
      self.phoning_err400_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0400', self.phoning_err400 )
      self.phoning_err560_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0560', self.phoning_err560 )
      self.phoning_err1000_summary = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_1000', self.phoning_err1000 )
    with tf.name_scope( "euler_error_posing" ):
      self.posing_err80   = tf.placeholder( tf.float32, name="posing_srnn_seeds_0080" )
      self.posing_err160  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0160" )
      self.posing_err320  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0320" )
      self.posing_err400  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0400" )
      self.posing_err560  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0560" )
      self.posing_err1000 = tf.placeholder( tf.float32, name="posing_srnn_seeds_1000" )

      self.posing_err80_summary   = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0080', self.posing_err80 )
      self.posing_err160_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0160', self.posing_err160 )
      self.posing_err320_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0320', self.posing_err320 )
      self.posing_err400_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0400', self.posing_err400 )
      self.posing_err560_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0560', self.posing_err560 )
      self.posing_err1000_summary = tf.summary.scalar( 'euler_error_posing/srnn_seeds_1000', self.posing_err1000 )
    with tf.name_scope( "euler_error_purchases" ):
      self.purchases_err80   = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0080" )
      self.purchases_err160  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0160" )
      self.purchases_err320  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0320" )
      self.purchases_err400  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0400" )
      self.purchases_err560  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0560" )
      self.purchases_err1000 = tf.placeholder( tf.float32, name="purchases_srnn_seeds_1000" )

      self.purchases_err80_summary   = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0080', self.purchases_err80 )
      self.purchases_err160_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0160', self.purchases_err160 )
      self.purchases_err320_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0320', self.purchases_err320 )
      self.purchases_err400_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0400', self.purchases_err400 )
      self.purchases_err560_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0560', self.purchases_err560 )
      self.purchases_err1000_summary = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_1000', self.purchases_err1000 )
    with tf.name_scope( "euler_error_sitting" ):
      self.sitting_err80   = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0080" )
      self.sitting_err160  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0160" )
      self.sitting_err320  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0320" )
      self.sitting_err400  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0400" )
      self.sitting_err560  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0560" )
      self.sitting_err1000 = tf.placeholder( tf.float32, name="sitting_srnn_seeds_1000" )

      self.sitting_err80_summary   = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0080', self.sitting_err80 )
      self.sitting_err160_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0160', self.sitting_err160 )
      self.sitting_err320_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0320', self.sitting_err320 )
      self.sitting_err400_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0400', self.sitting_err400 )
      self.sitting_err560_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0560', self.sitting_err560 )
      self.sitting_err1000_summary = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_1000', self.sitting_err1000 )
    with tf.name_scope( "euler_error_sittingdown" ):
      self.sittingdown_err80   = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0080" )
      self.sittingdown_err160  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0160" )
      self.sittingdown_err320  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0320" )
      self.sittingdown_err400  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0400" )
      self.sittingdown_err560  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0560" )
      self.sittingdown_err1000 = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_1000" )

      self.sittingdown_err80_summary   = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0080', self.sittingdown_err80 )
      self.sittingdown_err160_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0160', self.sittingdown_err160 )
      self.sittingdown_err320_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0320', self.sittingdown_err320 )
      self.sittingdown_err400_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0400', self.sittingdown_err400 )
      self.sittingdown_err560_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0560', self.sittingdown_err560 )
      self.sittingdown_err1000_summary = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_1000', self.sittingdown_err1000 )
    with tf.name_scope( "euler_error_takingphoto" ):
      self.takingphoto_err80   = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0080" )
      self.takingphoto_err160  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0160" )
      self.takingphoto_err320  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0320" )
      self.takingphoto_err400  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0400" )
      self.takingphoto_err560  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0560" )
      self.takingphoto_err1000 = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_1000" )

      self.takingphoto_err80_summary   = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0080', self.takingphoto_err80 )
      self.takingphoto_err160_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0160', self.takingphoto_err160 )
      self.takingphoto_err320_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0320', self.takingphoto_err320 )
      self.takingphoto_err400_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0400', self.takingphoto_err400 )
      self.takingphoto_err560_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0560', self.takingphoto_err560 )
      self.takingphoto_err1000_summary = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_1000', self.takingphoto_err1000 )
    with tf.name_scope( "euler_error_waiting" ):
      self.waiting_err80   = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0080" )
      self.waiting_err160  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0160" )
      self.waiting_err320  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0320" )
      self.waiting_err400  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0400" )
      self.waiting_err560  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0560" )
      self.waiting_err1000 = tf.placeholder( tf.float32, name="waiting_srnn_seeds_1000" )

      self.waiting_err80_summary   = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0080', self.waiting_err80 )
      self.waiting_err160_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0160', self.waiting_err160 )
      self.waiting_err320_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0320', self.waiting_err320 )
      self.waiting_err400_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0400', self.waiting_err400 )
      self.waiting_err560_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0560', self.waiting_err560 )
      self.waiting_err1000_summary = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_1000', self.waiting_err1000 )
    with tf.name_scope( "euler_error_walkingdog" ):
      self.walkingdog_err80   = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0080" )
      self.walkingdog_err160  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0160" )
      self.walkingdog_err320  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0320" )
      self.walkingdog_err400  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0400" )
      self.walkingdog_err560  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0560" )
      self.walkingdog_err1000 = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_1000" )

      self.walkingdog_err80_summary   = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0080', self.walkingdog_err80 )
      self.walkingdog_err160_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0160', self.walkingdog_err160 )
      self.walkingdog_err320_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0320', self.walkingdog_err320 )
      self.walkingdog_err400_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0400', self.walkingdog_err400 )
      self.walkingdog_err560_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0560', self.walkingdog_err560 )
      self.walkingdog_err1000_summary = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_1000', self.walkingdog_err1000 )
    with tf.name_scope( "euler_error_walkingtogether" ):
      self.walkingtogether_err80   = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0080" )
      self.walkingtogether_err160  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0160" )
      self.walkingtogether_err320  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0320" )
      self.walkingtogether_err400  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0400" )
      self.walkingtogether_err560  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0560" )
      self.walkingtogether_err1000 = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_1000" )

      self.walkingtogether_err80_summary   = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0080', self.walkingtogether_err80 )
      self.walkingtogether_err160_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0160', self.walkingtogether_err160 )
      self.walkingtogether_err320_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0320', self.walkingtogether_err320 )
      self.walkingtogether_err400_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0400', self.walkingtogether_err400 )
      self.walkingtogether_err560_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0560', self.walkingtogether_err560 )
      self.walkingtogether_err1000_summary = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_1000', self.walkingtogether_err1000 )

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

  def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs, forward_only, srnn_seeds=False):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs.
      decoder_inputs: list of numpy vectors to feed as decoder inputs.
      decoder_outputs: list of numpy vectors that are the expected decoder outputs.
      forward_only: whether to do the backward step or only forward.
      srnn_seeds: True if you want to evaluate using the sequences of SRNN
    Returns
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.
    Raises
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_inputs: decoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.keep_prob: 1.0}

    # Output feed: depends on whether we do a backward step or not.
    if not srnn_seeds:
      if not forward_only:

        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_inputs: decoder_inputs,
                      self.decoder_outputs: decoder_outputs,
                      self.keep_prob: 0.5}

        # Training step
        output_feed = [self.updates,         # Update Op that does SGD.
                       self.gradient_norms,  # Gradient norm.
                       self.training_loss,
                       self.loss_summary,
                       self.learning_rate_summary,
                       self.main_loss]

        outputs = session.run(output_feed, input_feed)
        
        return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]  # Gradient norm, loss, summaries

      else:
        # Validation step, not on SRNN's seeds
        output_feed = [self.validation_loss, # Loss for this batch.
                       self.loss_summary]

        outputs = session.run(output_feed, input_feed)
        
        return outputs[0], outputs[1]  # No gradient norm
    else:
      # Validation on SRNN's seeds
      output_feed = [self.validation_loss, # Loss for this batch.
                     self.outputs,
                     self.loss_summary]

      outputs = session.run(output_feed, input_feed)
      
      return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.


  def get_batch(self, data, actions):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

    for i in xrange(self.batch_size):

      the_key = all_keys[chosen_keys[i]]

      # Get the number of frames
      n, _ = data[the_key].shape
      # print(the_key, n)
      # Sample somewherein the middle
      idx = np.random.randint(0, n-total_frames)

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
      decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
      decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

    return encoder_inputs, decoder_inputs, decoder_outputs


  def find_indices_srnn(self, data, action):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = data[ (action, 1, 'downsampling') ].shape[0]


    idxs = []
    for i in range(0,8):
      idx = rng.randint(0, T1 - 75)
      idxs.append(idx)
    return idxs

  def get_batch_srnn(self, data, action ):
    """
    Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["walking","running","directing_traffic","soccer","basketball","washwindow","jumping","basketball_signal"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = self.find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 1
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs  = np.zeros((batch_size, source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)
    decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/c/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange(batch_size):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (action, subject, 'downsampling') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


    return encoder_inputs, decoder_inputs, decoder_outputs

def custom_encoder_net(cell, enc_in, num_layers, dropout, output_size, dtype=None, scope=None):
  """
  Custom encoder network

  Args
    cell: GRU cell
    enc_in: observed GT sequence
    num_layers: number of layers of RNN cell
    dropout: dropout probability
    output_size: output size of huamn pose
    scope: scope of encoder network

  Returns
    states: sequence of hidden states in encoder network
  """
  first_input = enc_in[0]
  input_shape = first_input.get_shape().with_rank_at_least(2)
  fixed_batch_size = input_shape[0]
  if fixed_batch_size.value:
    batch_size = fixed_batch_size.value
  else:
    batch_size = array_ops.shape(first_input)[0]
  state = cell.zero_state(batch_size,dtype)

  states = []
  outputs = []

  for enc in enc_in:
    # Add spatial attention
    # Get the variables from scope "spatial_attention"
    SA_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spatial_attention')
    SA_I_w = SA_list[0]
    SA_I_b = SA_list[1]
    SA_H_w = SA_list[2]
    SA_S_w = SA_list[3]
    SA_S_b = SA_list[4]

    # Compute the attention score for each frame
    enc_attention = tf.tanh(tf.matmul(enc, SA_I_w) + tf.matmul(state, SA_H_w) + SA_I_b)

    # Add FC layer and softmax
    enc_attention = tf.matmul(enc_attention,SA_S_w) + SA_S_b
    enc_attention = tf.nn.softmax(enc_attention)

    # apply score
    enc = tf.multiply(enc_attention, enc)

    _, state = cell(enc, state)
    states.append(state)

  return states

def custom_decoder_net(dec_in, enc_states, cell, length_in, length_out, num_layers, dropout, output_size, sub_idx, scope):
  """
  Custom RNN network to grab all states
  Args
    dec_in: input sequence to decoder
    enc_state: Hidden state of RNN from encoder part
    cell: RNN cell with linear encoder
    length_in: length of input
    length_out: length of output
    dropout: probability
    scope: variables scope
  Return
    outputs: list of outputs
    states: list of states
  """
  states, outputs, velocities = [], [], []
  # Last encoder hidden states
  enc_state = enc_states[-1]
  # batch size
  batch_size = tf.shape(enc_state)[0]

  # hs = tf.transpose(enc_states, [1, 0, 2])

  # with tf.variable_scope(scope):
  for i in range(length_out):
    # save input for residual connection
    if i > 0:
      dec_input = outputs[-1]
      state_input = states[-1]
    else:
      dec_input = dec_in[0]
      state_input = enc_state

    # Add spatial attention
    # Get the variables from scope "spatial_attention"
    if sub_idx == -1:
      SA_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='spatial_attention')
    else:
      SA_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sub_spatial_attention')

    SA_I_w = SA_list[0]
    SA_I_b = SA_list[1]
    SA_H_w = SA_list[2]
    SA_S_w = SA_list[3]
    SA_S_b = SA_list[4]

    # Compute the attention score for each frame
    dec_attention = tf.tanh(tf.matmul(dec_input, SA_I_w) + tf.matmul(state_input, SA_H_w) + SA_I_b)

    # Add FC layer and softmax
    dec_attention = tf.matmul(dec_attention,SA_S_w) + SA_S_b
    dec_attention = tf.nn.softmax(dec_attention)

    # apply score
    dec = tf.multiply(dec_attention,dec_input)
    output, state = cell(dec, state_input)

    # Add dropout and FC layer
    output = tf.contrib.layers.dropout(output, dropout)
    if i == 0 and sub_idx <= 0:
      output = tf.contrib.layers.fully_connected(output, 512, activation_fn=tf.nn.relu, reuse=None, scope = scope + "_fully_connected_1")
      output = tf.contrib.layers.dropout(output, dropout)
      output = tf.contrib.layers.fully_connected(output, output_size, activation_fn=None, reuse=None, scope= scope + "_fully_connected_2")
    else:
      output = tf.contrib.layers.fully_connected(output, 512, activation_fn=tf.nn.relu, reuse=True, scope= scope + "_fully_connected_1")
      output = tf.contrib.layers.dropout(output, dropout)
      output = tf.contrib.layers.fully_connected(output, output_size, activation_fn=None, reuse=True, scope=scope + "_fully_connected_2")

    # residual connection
    output = tf.add(output, dec_input)

    outputs.append(output)
    states.append(state)
  return outputs, states

