/**
 * Compute binomial coefficients C_n^k = n! / k! / (n-k)!
 *
 * Adapted from http://go-lambda.blogspot.fr/2012/02/template-for-binomial-coefficients-in-c.html
 */
#ifndef BINOMIAL_H_
#define BINOMIAL_H_

namespace mood {

template<int n, int k>
struct Binomial
{
  const static int value =  (Binomial<n-1,k-1>::value + Binomial<n-1,k>::value);
};

template<>
struct Binomial<0,0>
{
  const static int value = 1;
};

template<int n>
struct Binomial<n,0>
{
  const static int value = 1;
};

template<int n>
struct Binomial<n,n>
{
  const static int value = 1;
};

template<int n, int k>
inline constexpr int binomial()
{
  return Binomial<n,k>::value;
}

/**
 * binomial coefficients without templates.
 */
inline int binom(int n, int k)
{

  if (n < k) return 0;
  if (k == 0 || n == 1) return 1;
  if (n == 2 && k == 1) return 2;
  if (n == 2 && k == 2) return 1;
  if (n == k) return 1;

  int res = 1;

  if ( k > n - k ) k = n - k;
  for( int i = 0; i < k; ++i ) {
    res *= ( n - i );
    res /= ( i + 1 );
  }
  return res;
  
} // binom

} // namespace mood

#endif // BINOMIAL_H_
