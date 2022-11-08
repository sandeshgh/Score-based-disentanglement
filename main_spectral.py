# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
import run_lib_spectral

# print('debugging, imported stuffs')
os.environ["CUDA_DEVIC_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "visualize", "analyze", "analyze_close", "plot_data"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  print('reached main')
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib_spectral.train(FLAGS.config, FLAGS.workdir)
    ####
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib_spectral.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  elif FLAGS.mode == "visualize":
    # Run visualization
    run_lib_spectral.visualize(FLAGS.config, FLAGS.workdir, "vizual")
  elif FLAGS.mode == "analyze":
    # Run the evaluation pipeline
    run_lib_spectral.analyze(FLAGS.config, FLAGS.workdir, "analyze")
  elif FLAGS.mode == "analyze_close":
    # Run the evaluation pipeline
    run_lib_spectral.analyze_close(FLAGS.config, FLAGS.workdir, "analyze_close")
  elif FLAGS.mode == "plot_data":
    # Run the evaluation pipeline
    run_lib_spectral.plot_data(FLAGS.config, FLAGS.workdir, "dataplot")
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
