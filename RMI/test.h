#include <cstddef>
#include <cstdint>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
typedef boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float< 512, boost::multiprecision::backends::digit_base_2, void, boost::int16_t, -16382, 16383>,boost::multiprecision::et_off>  cpp_bin_float_512;
namespace test {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 6442450952;
const uint64_t BUILD_TIME_NS = 674328401871;
const char NAME[] = "test";
uint64_t lookup(uint64_t key, size_t* err);
}
