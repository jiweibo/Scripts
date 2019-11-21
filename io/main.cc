#include "io.h"
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(file, "test.txt", "input file txt");
DEFINE_int32(batch_size, 1, "the batch size");
DEFINE_int32(input_size, 4, "the col of txt");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TxtDataReader reader(FLAGS_file, FLAGS_batch_size, FLAGS_input_size);
  std::vector<std::vector<float> > data;
  while(reader.get_next_batch(&data, ';', ' ')) {
    std::cout << "batch: -------------- " << std::endl;
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[i].size(); ++j) {
        std::cout << data[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }
}