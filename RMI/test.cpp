#include "test.h"
#include "test_data.h"
#include <math.h>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
namespace test {
bool load(char const* dataPath) {
  {
    using boost::multiprecision::cpp_int;
    std::ifstream infile(std::filesystem::path(dataPath) / "test_L0_PARAMETERS", std::ios::in | std::ios::binary);
    if (!infile.good()) return false;
    infile.read((char*)L0_PARAMETERS, 16);
    
  }
  {
    using boost::multiprecision::cpp_int;
    std::ifstream infile(std::filesystem::path(dataPath) / "test_L1_PARAMETERS", std::ios::in | std::ios::binary);
    if (!infile.good()) return false;
    L1_PARAMETERS = (char*) malloc(6442450944);
    if (L1_PARAMETERS == NULL) return false;
    infile.read((char*)L1_PARAMETERS, 6442450944);
    
  }
  return true;
}
void cleanup() {
    free(L1_PARAMETERS);
}

inline uint64_t pwl(uint64_t kmer, uint64_t inp) {

    return inp >> (64-kmer);
}

inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}

inline size_t FCLAMP(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}

uint64_t lookup(uint64_t key, size_t* err) {
  double fpred;
  uint64_t ipred;
  size_t modelIndex;
  ipred = pwl(L0_PARAMETERS[2*0 + 0], (uint64_t)key);
  modelIndex = (ipred > 268435456 - 1 ? 268435456 - 1 : ipred);
  fpred = linear(*((double*) (L1_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L1_PARAMETERS + (modelIndex * 24) + 8)), (double)key);
  *err = *((uint64_t*) (L1_PARAMETERS + (modelIndex * 24) + 16));

  return FCLAMP(fpred, 6203609478.0 - 1.0);
}
} // namespace
