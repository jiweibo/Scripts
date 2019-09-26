import paddle
import argparse
import numpy as np
import paddle.fluid as fluid

def parse_args():
  parser = argparse.ArgumentParser(description='paddle activation op test')
  parser.add_argument("op", type=str, choices=['leaky_relu'], help="activation op in 'leaky_relu', default is 'leaky_relu'", default='leaky_relu')
  args = parser.parse_args()
  return args, parser

def main():
  x = fluid.layers.data(name='x', shape=[2], dtype='float32', lod_level=1)
  result = fluid.layers.leaky_relu(x, alpha=0.1)
 
  exe = fluid.Executor(fluid.CPUPlace())
  exe.run(fluid.default_startup_program())
  
  x_i = np.array([[-1, 2], [3, -4]]).astype(np.float32)
  x_d = fluid.create_lod_tensor(x_i, [[1, 1]], fluid.CPUPlace())
  
  result_value, = exe.run(fluid.default_main_program(), feed={'x':x_d}, fetch_list=[result], return_numpy=False)
  #print(type(result_value))
  print(result_value)

if __name__ == '__main__':
  args, parser = parse_args()
  #parser.print_help()
  main()
