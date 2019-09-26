import paddle
import argparse
import numpy as np
import paddle.fluid as fluid

def parse_args():
  parser = argparse.ArgumentParser(description='paddle compare op test')
  parser.add_argument("op", type=str, choices=['less_than', 'less_equal'], help="compare op in 'less_than', 'less_equal', default is 'less_than'", default='less_than')
  args = parser.parse_args()
  return args, parser

def main():
  x = fluid.layers.data(name='x', shape=[2], dtype='float64', lod_level=1)
  y = fluid.layers.data(name='y', shape=[2], dtype='float64', lod_level=1)
  # ref = fluid.layers.fill_constant(shape=[2], dtype='float64', value=0)
  result = fluid.layers.less_than(x=x, y=y, force_cpu=False) 
 
  place = fluid.CPUPlace()
  
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())
  
  x_i = np.array([[1, 2], [3, 4]]).astype(np.float64)
  y_i = np.array([[2, 2], [1, 3]]).astype(np.float64)

  x_d = fluid.create_lod_tensor(x_i, [[1,1]], place)
  y_d = fluid.create_lod_tensor(y_i, [[1,1]], place)

  result_value, = exe.run(fluid.default_main_program(), feed={'x':x_d, 'y':y_d}, fetch_list=[result], return_numpy=False)
  #print(type(result_value))
  print(result_value)
  print(np.array(result_value))
  #print(ref_value)

if __name__ == '__main__':
  args, parser = parse_args()
  #parser.print_help()
  main()
