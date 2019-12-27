#include "timer.h"

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {

  auto start = time();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  auto end = time();

  std::cout << "time is " << time_diff(start, end) << std::endl;
  return 0;
}