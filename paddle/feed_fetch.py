'''
A separate Fluid test file for feeding specific data
'''

import sys
import os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core
import subprocess
import argparse


def parse_args():
    '''
    parse args
    '''
    parser = argparse.ArgumentParser(
        description='Fluid test file for feeding specific data')
    parser.add_argument('model_dir', type=str, help='model dir')
    parser.add_argument('--model_name', type=str,
                        help='model name', default='model')
    parser.add_argument('--param_name', type=str,
                        help='param name', default='param')
    parser.add_argument('--arg_name', type=str, help='arg name', default=None)
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--draw', dest='draw',
                        action='store_true', help='draw model info')
    parser.add_argument('--save', dest='save',
                        action='store_true', help='save result in txt')
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


def feed_ones(block, feed_target_names, batch_size=1):
    '''
    '''
    feed_dict = dict()

    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        print(shape)
        return shape

    def fill_ones(var_name, batch_size):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64
        }
        print(var_name, var.dtype)
        np_dtype = var_np[var.dtype]
        return np.ones(np_shape, dtype=np_dtype)
    for feed_target_name in feed_target_names:
        print('feed_name: ', feed_target_name)
        feed_dict[feed_target_name] = fill_ones(feed_target_name, batch_size)
    return feed_dict


def fetch_tmp_vars(block, fetch_targets, var_names_list=None):
    '''
    '''
    def var_names_of_fetch(fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list

    fetch_var = block.var('fetch')
    old_fetch_names = var_names_of_fetch(fetch_targets)
    new_fetch_vars = []
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)
    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()
    for var_name in var_names_list:
        if '.tmp_' in var_name and var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(type='fetch', inputs={'X': [var_name]}, outputs={
                            'Out': [fetch_var]}, attrs={'col': i})
            i += 1
    return new_fetch_vars


def print_results(results, fetch_targets, need_save=False):
    '''
    '''
    for result in results:
        idx = results.index(result)
        var = fetch_targets[idx]
        print(var.name, ' shape is: ', str(var.shape),
              ' mean value is: ', np.mean(np.array(result)))
        if need_save is True:
            numpy_to_txt(result, 'result_' +
                         fetch_targets[idx].name.replace('/', ''), True)


def numpy_to_txt(numpy_array, save_name, print_shape=True):
    '''
    transform numpy to txt
    '''
    np_array = np.array(numpy_array)
    fluid_fetch_list = list(np_array.flatten())
    fetch_txt_fp = open(save_name + '.txt', 'w')
    for num in fluid_fetch_list:
        fetch_txt_fp.write(str(num) + '\n')
    if print_shape is True:
        fetch_txt_fp.write('Shape: (')
        for val in np_array.shape:
            fetch_txt_fp.write(str(val) + ', ')
        fetch_txt_fp.write(')\n')
    fetch_txt_fp.close()


def fluid_inference_test(model_path, model_name, param_name, arg_name, drawpdf=False, save=False):
    '''
    '''
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        [inference_program, feed_names, fetch_targets] = load_inference_model(
            model_path, model_name, param_name, exe)
        global_block = inference_program.global_block()
        if (drawpdf):
            draw(global_block)
            # print(global_block.vars.keys())
        feed_list = feed_ones(global_block, feed_names, 1)
        fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [arg_name])
        # print(fetch_targets)
        results = exe.run(program=inference_program, feed=feed_list,
                          fetch_list=fetch_targets, return_numpy=False)
        print_results(results, fetch_targets, need_save=save)


if __name__ == '__main__':
    args, parser = parse_args()
    fluid_inference_test(args.model_dir, args.model_name,
                         args.param_name, args.arg_name, args.draw, args.save)
