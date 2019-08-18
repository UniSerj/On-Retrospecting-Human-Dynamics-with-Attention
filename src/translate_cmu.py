
"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils_cmu as data_utils
import seq2seq_model_cmu as seq2seq_model

# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", 100000, "Iterations to train for.")
tf.app.flags.DEFINE_float("loss_weight_decay_factor", 1.0, "Weight for mean_sub_loss is multiplied by this much")
tf.app.flags.DEFINE_integer("loss_weight_step", 2000, "Every this many steps, do decay.")
# Architecture
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 25, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("omit_one_hot", False, "Whether to remove one-hot encoding from the data")
tf.app.flags.DEFINE_integer("reconstruction_len", 6, "Number of frames to review before anchor points")
tf.app.flags.DEFINE_integer("anchor_len", 6, "Number of frames to between anchor points")
tf.app.flags.DEFINE_float("sub_loss_weight", 0.5, "Weight for mean_sub_loss")
# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("../data/cmu_mocap/"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")

tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")

tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_integer("experiments", 1, "Experiments id.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.normpath(os.path.join(FLAGS.train_dir, FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'experiments_{0}'.format(FLAGS.experiments)))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

def create_model(session, actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.seq_length_in if not sampling else 50,
      FLAGS.seq_length_out if not sampling else 25,
      FLAGS.size, # hidden layer size
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      summaries_dir,
      len(actions),
      not FLAGS.omit_one_hot,
      FLAGS.reconstruction_len,
      FLAGS.anchor_len,
      FLAGS.sub_loss_weight,
      FLAGS.loss_weight_decay_factor,
      dtype=tf.float32)

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) ))
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model


def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions(FLAGS.action)

  number_of_actions = len(actions)

  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count = device_count)) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    model = create_model(sess, actions)
    model.train_writer.add_graph(sess.graph)
    print("Model created")

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_euler = get_srnn_gts(actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot)

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0

    # Init Validation variable for the whole model
    minimized_error = 0.0
    recorded_step = 0

    for _ in xrange(FLAGS.iterations):

      start_time = time.time()

      # === Training step ===
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, not FLAGS.omit_one_hot )
      _, step_loss, loss_summary, lr_summary, main_loss = model.step( sess, encoder_inputs,
       decoder_inputs, decoder_outputs, False )

      model.train_writer.add_summary(loss_summary, current_step)
      model.train_writer.add_summary(lr_summary, current_step)

      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

      step_time += (time.time() - start_time) / FLAGS.test_every
      loss += step_loss / FLAGS.test_every
      current_step += 1

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)
      if current_step % FLAGS.loss_weight_step == 0:
        sess.run(model.sub_loss_weight_decay_op)

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.test_every == 0:

        # === Validation with randomly chosen seeds ===
        forward_only = True

        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set, not FLAGS.omit_one_hot)
        step_loss, loss_summary = model.step(sess,
            encoder_inputs, decoder_inputs, decoder_outputs, forward_only)

        val_loss = step_loss # Loss book-keeping

        model.test_writer.add_summary(loss_summary, current_step)

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()

        # Init the variable for accmulated error
        accumulated_error = 0.0
        mean_mean_errors_avg = {}
        for i in range(0, 25):
          mean_mean_errors_avg[i] = 0


        # === Validation with srnn's seeds ===
        for action in actions:

          # Evaluate the model on the test batches
          encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(test_set, action)
          srnn_loss, srnn_poses, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                   decoder_outputs, True, True)

          # Denormalize the output
          srnn_pred_expmap = data_utils.revert_output_format(srnn_poses,
            data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot)

          # Save the errors here
          mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))

          # Training is done in exponential map, but the error is reported in
          # Euler angles, as in previous work.
          # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
          N_SEQUENCE_TEST = 8
          for i in np.arange(N_SEQUENCE_TEST):
            eulerchannels_pred = srnn_pred_expmap[i]

            # Convert from exponential map to Euler angles
            for j in np.arange(eulerchannels_pred.shape[0]):
              for k in np.arange(3,115,3):
                eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                  data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

            # The global translation (first 3 entries) and global rotation
            # (next 3 entries) are also not considered in the error, so the_key
            # are set to zero.
            # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
            gt_i=np.copy(srnn_gts_euler[action][i])
            gt_i[:,0:6] = 0

            # Now compute the l2 error.
            idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

            euc_error = np.power(gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
            euc_error = np.sum(euc_error, 1)
            euc_error = np.sqrt(euc_error)
            mean_errors[i,:] = euc_error

          # This is simply the mean error over the N_SEQUENCE_TEST examples
          mean_mean_errors = np.mean(mean_errors, 0)
          accumulated_error += np.mean(mean_mean_errors)

          for ms in range(0, 25):
            mean_mean_errors_avg[ms] += mean_mean_errors[ms]

          # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
          print("{0: <16} |".format(action), end="")
          for ms in [1,3,7,9,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
              # mean_mean_errors_avg[ms] += mean_mean_errors[ms]
            else:
              print("   n/a |", end="")
          print()

        for i in range(0, FLAGS.seq_length_out):
          mean_mean_errors_avg[i] = mean_mean_errors_avg[i]/number_of_actions

        # Pretty print of the average results for 80, 160, 320, 400, 560 and 1000 ms
        print("{0: <16} |".format("average"), end="")
        for ms in [1,3,7,9,24]:
          if FLAGS.seq_length_out >= ms+1:
            print(" {0:.3f} |".format( mean_mean_errors_avg[ms] ), end="")
          else:
            print("   n/a |", end="")
        print()


        print()
        print("============================\n"
              "Global step:         %d\n"
              "Learning rate:       %.4f\n"
              "Step-time (ms):     %.4f\n"
              "Train loss avg:      %.4f\n"
              "--------------------------\n"
              "Val loss:            %.4f\n"
              "srnn loss:           %.4f\n"
              "============================" % (model.global_step.eval(),
              model.learning_rate.eval(), step_time*1000, loss,
              val_loss, srnn_loss))
        print()

        # Update minimized_error and recorded step
        if (accumulated_error < minimized_error) or (minimized_error == 0.0):
          minimized_error = accumulated_error
          recorded_step = model.global_step.eval()
          print("Updated Error:{0}, step:{1}".format(minimized_error,recorded_step))

          # Save the model
          if current_step % FLAGS.save_every == 0:
            print("Saving the model..." ); start_time = time.time()
            model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step)
            print("done in {0:.2f} ms".format( (time.time() - start_time)*1000))

        else:
          print("Error:{0}".format(accumulated_error))

        previous_losses.append(loss)

        # Reset global time and loss
        step_time, loss = 0, 0

        sys.stdout.flush()

    print()
    print("Minimized_error:{0}, on step:{1}".format(minimized_error,recorded_step))
    return minimized_error,recorded_step


def get_srnn_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, _, srnn_expmap = model.get_batch_srnn( test_set, action )
    # print(srnn_expmap.shape)
    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,115,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def get_srnn_gts_sample( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, to_euler=True ):
  """
  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    encoder_input, decoder_input, decoder_output = model.get_batch_srnn( test_set, action )
    # print(np.array(encoder_input).shape,np.array(decoder_input).shape,np.array(decoder_output).shape)
    srnn_expmap  = np.zeros( (8, 25+4, 70), dtype=float )
    for i in np.arange( decoder_output.shape[0] ):
      sequence = np.concatenate((encoder_input[i][-3:],decoder_input[i][0:1], decoder_output[i][:]),axis=0)
      srnn_expmap[i, :, :] = sequence[:,:]
    # print(srnn_expmap.shape)

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,115,3):
            denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed );

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler

def sample():
  """Sample predictions for srnn's seeds"""

  if FLAGS.load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions(FLAGS.action)

  # Use the CPU if asked to
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(device_count = device_count)) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    sampling     = True
    model = create_model(sess, actions, sampling)
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
      actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts_sample( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False )
    srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot )

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
      os.remove( SAMPLES_FNAME )
    except OSError:
      pass

    # Predict and save for each action
    for action in actions:

      # Make prediction with srnn' seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action )
      forward_only = True
      srnn_seeds = True
      srnn_loss, srnn_poses, _, spatial_weight = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, forward_only, srnn_seeds)

      # denormalizes too
      srnn_pred_expmap = data_utils.revert_output_format( srnn_poses, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

      # Save the conditioning seeds

      # Save the samples
      with h5py.File(SAMPLES_FNAME, 'a') as hf:
        for i in np.arange(8):
          # Save conditioning ground truth
          node_name = 'expmap/gt/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_gts_expmap[action][i] )
          # Save prediction
          node_name = 'expmap/preds/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

      for i in np.arange(8):

        eulerchannels_pred = srnn_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,115,3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat(eulerchannels_pred[j,k:k+3]))

        eulerchannels_pred[:,0:6] = 0

        # Pick only the dimensions with sufficient standard deviation. Others are ignored.
        idx_to_use = np.where(np.std(eulerchannels_pred, 0) > 1e-4)[0]

        euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean(mean_errors, 0)
      print(action)
      print(','.join(map(str, mean_mean_errors.tolist())))

      with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
        node_name = 'mean_{0}_error'.format(action)
        hf.create_dataset(node_name, data=mean_mean_errors)
  return


def define_actions( action ):
  """
  Define the list of actions we are using.

  Args
    action: String with the passed action. Could be "all"
  Returns
    actions: List of strings of actions
  Raises
    ValueError if the action is not included in H3.6M
  """

  actions = ["walking","running","directing_traffic","soccer","basketball","washwindow","jumping","basketball_signal"]

  if action in actions:
    return [action]

  if action == "all":
    return actions

  raise( ValueError, "Unrecognized action: %d" % action )


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """
  Loads data for training/testing and normalizes it.

  Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
  Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))

  # train_subject_ids = [1,6,7,8,9,11]
  # test_subject_ids = [5]

  train_set, complete_train = data_utils.load_data(data_dir+'/train', actions)
  test_set,  complete_test  = data_utils.load_data(data_dir+'/test', actions)

  # Compute normalization stats
  data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

  # Normalize -- subtract mean, divide by stdev
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")
  # print(len(dim_to_use),train_set.keys())

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()


if __name__ == "__main__":
  tf.app.run()
