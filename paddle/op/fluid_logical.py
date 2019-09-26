import paddle
import argparse
import numpy as np
import paddle.fluid as fluid

def parse_args():
  parser = argparse.ArgumentParser(description='paddle logical op test')
  parser.add_argument("op", type=str, choices=['and', 'or', 'xor', 'not'], help="logical op in 'and', 'or', 'xor', 'not', default is 'and'", default='and')
  args = parser.parse_args()
  return args, parser

def main():
  x = fluid.layers.data(name='x', shape=[1], dtype='bool')
  y = fluid.layers.data(name='y', shape=[1], dtype='bool')
  op = args.op
  if (op == "and"):
    result = fluid.layers.logical_and(x=x, y=y)
  elif (op == "or"):
    result = fluid.layers.logical_or(x=x, y=y)
  elif (op == "xor"):
    result = fluid.layers.logical_xor(x=x, y=y)
  else:
    result = fluid.layers.logical_not(x=x)
  
  place = fluid.CPUPlace()
  
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())
  
  x_i = np.ones((10, 1)).astype(np.bool)
  y_i = np.zeros((10, 1)).astype(np.bool)
  
  result_value, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[result])
  print(type(result_value))
  print(result_value)

if __name__ == '__main__':
  args, parser = parse_args()
  parser.print_help()
  main()
