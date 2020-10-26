#include "paddle/include/paddle_inference_api.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(model_dir, "./mobilenetv1", "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_int32(thread_num, 1, "thread num");
DEFINE_int32(math_num, 1, "SetCpuMathLibraryNumThreads");
DEFINE_bool(profile, false, "native profile");
DEFINE_bool(mkldnn, false, "enable mkldnn");

namespace paddle {

using paddle::AnalysisConfig;

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds us;
  auto diff = t2 - t1;
  us counter = std::chrono::duration_cast<us>(diff);
  return counter.count() / 1000.0;
}

void PrepareConfig(AnalysisConfig *config) {
  config->SetModel(FLAGS_model_dir + "/model", FLAGS_model_dir + "/params");
  config->DisableGpu();

  if (FLAGS_mkldnn) {
    config->EnableMKLDNN();
  }

  config->SetCpuMathLibraryNumThreads(FLAGS_math_num);

  // We use ZeroCopyTensor here, so we set config->SwitchUseFeedFetchOps(false)
  config->SwitchUseFeedFetchOps(false);

  if (FLAGS_profile) {
    config->EnableProfile();
  }
}

void work_thread(PaddlePredictor *predictor_master, int thread_id) {
  auto predictor = predictor_master->Clone();

  int batch_size = 1;
  int channels = 3;
  int height = 224;
  int width = 224;
  int input_num = channels * height * width * batch_size;

  // prepare inputs
  float *input = new float[input_num];
  memset(input, 0, input_num * sizeof(float));

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  for (size_t i = 0; i < FLAGS_warmup; ++i) {
    CHECK(predictor->ZeroCopyRun());
  }

  auto time1 = time();
  for (size_t i = 0; i < FLAGS_repeats; i++) {
    CHECK(predictor->ZeroCopyRun());
  }
  auto time2 = time();
  LOG(INFO) << "thread_id: " << thread_id << " batch: " << batch_size
            << " predict cost: "
            << time_diff(time1, time2) / static_cast<float>(FLAGS_repeats)
            << "ms" << std::endl;

  // get the output
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
  for (size_t j = 0; j < out_num; ++j) {
    // LOG(INFO) << "output[" << j << "]: " << out_data[j];
  }
  delete[] input;
}
} // namespace paddle

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  paddle::AnalysisConfig config;
  PrepareConfig(&config);
  auto predictor = paddle::CreatePaddlePredictor(config);

  std::vector<std::thread> threads;

  for (int i = 0; i < FLAGS_thread_num; i++) {
    threads.push_back(std::thread(paddle::work_thread, predictor.get(), i));
  }
  for (auto &t : threads) {
    t.join();
  }
  return 0;
}
