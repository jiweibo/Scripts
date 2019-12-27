#include <chrono>

using Time = decltype(std::chrono::high_resolution_clock::now());
inline Time time() { return std::chrono::high_resolution_clock::now(); };
inline double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds us;
  auto diff = t2 - t1;
  us counter = std::chrono::duration_cast<us>(diff);
  return counter.count() / 1000.0;
}
