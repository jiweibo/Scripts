import sys
import numpy as np
import os
import argparse
import paddle
paddle.enable_static()
import paddle.fluid as fluid
from paddle.fluid import core
import subprocess

def parse_args():
  parser = argparse.ArgumentParser(description='prune paddle models')
  parser.add_argument('src_dir', type=str, help='source model dir')
  parser.add_argument('dst_dir', type=str, help='dst model dir')
  parser.add_argument("feed_names", type=str, help="feed names")
  parser.add_argument("fetch_names", type=str, help="fetch names")
  parser.add_argument('--src_model_name', type=str, default='inference.pdmodel', help='model filename')
  parser.add_argument("--src_params_name", type=str, default='inference.pdiparams', help="params filename")
  parser.add_argument('--dst_model_name', type=str, default='inference.pdmodel', help='model filename')
  parser.add_argument("--dst_params_name", type=str, default='inference.pdiparams', help="params filename")
  args = parser.parse_args()
  return args, parser

def load_inference_model(model_path, model_name, param_name, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, model_name)
    param_abs_path = os.path.join(model_path, param_name)
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, model_name, param_name)
    else:
        return fluid.io.load_inference_model(model_path, exe)

def prune(program, feeds, fetches):
    global_block = program.global_block()
    target_vars = [global_block.var(i) for i in fetches]
    new_program = program._prune_with_input(
           feeded_var_names=feeds, targets=target_vars)
    fluid.io.prepend_feed_ops(new_program, feeds)
    fluid.io.append_fetch_ops(new_program, fetches)
    return new_program

def prune_inference_program(load_path, src_model_name, src_params_name, save_path, dst_model_name, dst_params_name, feed_names, fetch_names):
    """
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        [net_program,
        feed_target_names,
        fetch_targets] = load_inference_model(load_path, src_model_name, src_params_name, exe)
        fetched_vars = [net_program.global_block().var(x) for x in fetch_names]
        new_program = prune(net_program, feed_names, fetch_names)

        if (not os.path.exists(save_path)):
            os.makedirs(save_path)
        
        #new_program.global_block()._remove_var('save_infer_model/scale_0')
        fluid.io.save_inference_model(save_path, feeded_var_names=feed_names, target_vars=fetched_vars, executor=exe, main_program=new_program, model_filename=dst_model_name, params_filename=dst_params_name)
        #with open(os.path.join(save_path, '__model__'), "wb+") as f:
        #    f.write(new_program.desc.serialize_to_string())
        #fluid.io.save_persistables(exe, save_path, net_program)

if __name__ == "__main__":
  args, parser = parse_args()
  feed_names = args.feed_names.split(':')
  fetch_names = args.fetch_names.split(':')
  
  prune_inference_program(args.src_dir, args.src_model_name, args.src_params_name, args.dst_dir, args.dst_model_name, args.dst_params_name, feed_names, fetch_names)

