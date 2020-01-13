import argparse
from google.protobuf import text_format
import paddle.fluid as fluid

from paddle.fluid.proto.framework_pb2 import ProgramDesc

def parse_args():
  '''
  parse args
  '''
  parser = argparse.ArgumentParser(description='Fluid model to text file')
  parser.add_argument("path", type=str, help='__model__ bin path or __model__ text path')
  parser.add_argument('--file', type=str, help='save in txt', default='model')
  parser.add_argument('--serialize', dest='serialize', action='store_true', help='save new__model__')
  args = parser.parse_args()
  return args, parser

if __name__ == '__main__':
  args, parser = parse_args()
  print('''usage:\n1.: bin2txt, python bin2txt.py __model__ --file=model.txt\n2.: txt2bin, python bin2txt.py model.txt --file=__model__new\n''')
  program = ProgramDesc()
  if not args.serialize:
    f = open(args.path, 'rb')
    program.ParseFromString(f.read())
    f.close()
    with open(args.file, 'w') as f:
      f.write(text_format.MessageToString(program))
  else:
    f = open(args.path, 'r')
    text_format.Parse(f.read(), program)
    f.close()
    f = open(args.file, 'wb')
    f.write(program.SerializeToString())
    f.close()
