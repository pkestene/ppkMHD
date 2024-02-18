/*
 * http://stackoverflow.com/questions/2940367/what-is-more-efficient-using-pow-to-square-or-just-multiply-it-with-itself
 *
 *
 * This test shows that one can use pow(base, exponent) when exponent is an integer,
 * it is as fast as performing multiplications "manually".
 *
 * For CUDA, this is no pow(double,int) interface, so we need to use the manual
 * multiplications (templated macros).
 *
 * g++ -O3 -o test_pow test_pow.cpp
 */

#include <cstdlib>
#include <cmath>
#include <boost/date_time/posix_time/posix_time.hpp>

inline boost::posix_time::ptime
now()
{
  return boost::posix_time::microsec_clock::local_time();
}

#define TEST(num, expression)	       \
  double test##num(double b, long loops)	\
  {						\
    double x = 0.0;				\
						\
    boost::posix_time::ptime startTime = now(); \
    for (long i=0; i<loops; ++i)		\
      {						\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
        x += expression;			\
      }								  \
    boost::posix_time::time_duration elapsed = now() - startTime; \
								  \
    std::cout << elapsed << " ";				  \
								  \
    return x;							  \
  }

TEST(1, b)
TEST(2, b * b)
TEST(3, b * b * b)
TEST(4, b * b * b * b)
TEST(5, b * b * b * b * b)

template <int exponent>
double
testpow(double base, long loops)
{
  double x = 0.0;

  boost::posix_time::ptime startTime = now();
  for (long i = 0; i < loops; ++i)
  {
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
    x += std::pow(base, exponent);
  }
  boost::posix_time::time_duration elapsed = now() - startTime;

  std::cout << elapsed << " ";

  return x;
}

template <int exponent>
double
testpow_double(double base, long loops)
{
  double x = 0.0;

  boost::posix_time::ptime startTime = now();
  for (long i = 0; i < loops; ++i)
  {
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
    x += std::pow(base, (double)exponent);
  }
  boost::posix_time::time_duration elapsed = now() - startTime;

  std::cout << elapsed << " ";

  return x;
}

int
main()
{
  using std::cout;
  long loops = 100000000l;

  cout << "###############################################\n";
  cout << "Compare std::pow(double, int) with manual power\n";
  cout << "###############################################\n";
  double x = 0.0;

  cout << "1 ";
  x += testpow<1>(rand(), loops);
  x += test1(rand(), loops);

  cout << "\n2 ";
  x += testpow<2>(rand(), loops);
  x += test2(rand(), loops);

  cout << "\n3 ";
  x += testpow<3>(rand(), loops);
  x += test3(rand(), loops);

  cout << "\n4 ";
  x += testpow<4>(rand(), loops);
  x += test4(rand(), loops);

  cout << "\n5 ";
  x += testpow<5>(rand(), loops);
  x += test5(rand(), loops);
  cout << "\n" << x << "\n";

  cout << "##################################################\n";
  cout << "Compare std::pow(double, double) with manual power\n";
  cout << "##################################################\n";
  x = 0.0;
  loops = 4000000l;

  cout << "1 ";
  x += testpow_double<1>(rand(), loops);
  x += test1(rand(), loops);

  cout << "\n2 ";
  x += testpow_double<2>(rand(), loops);
  x += test2(rand(), loops);

  cout << "\n3 ";
  x += testpow_double<3>(rand(), loops);
  x += test3(rand(), loops);

  cout << "\n4 ";
  x += testpow_double<4>(rand(), loops);
  x += test4(rand(), loops);

  cout << "\n5 ";
  x += testpow_double<5>(rand(), loops);
  x += test5(rand(), loops);
  cout << "\n" << x << "\n";
}
