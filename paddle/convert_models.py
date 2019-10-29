################################
# save paddle inference model
################################

import paddle.fluid as fluid
import numpy as np
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='paddle convert inference models.')
  parser.add_argument("dirname", type=str, help="source model dir")
  parser.add_argument("model_filename", type=str, help="model filename")
  parser.add_argument("params_filename", type=str, help="params filename")
  parser.add_argument("save_dir", type=str, help="target model dir")
  args = parser.parse_args()
  return args, parser

def convert():
  exe = fluid.Executor(fluid.CPUPlace())
  
  [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=args.dirname, executor=exe, model_filename=args.model_filename, params_filename=args.params_filename)
  
  with fluid.program_guard(inference_program):
    fluid.io.save_inference_model(args.save_dir, feeded_var_names=feed_target_names, target_vars=fetch_targets, executor=exe)

if __name__ == '__main__':
  args, parser = parse_args()
  convert()
