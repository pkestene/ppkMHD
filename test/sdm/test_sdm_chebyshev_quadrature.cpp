/**
 * This simple executable is there only to make sure there is no
 * mistake in performing a Gauss-Chebyshev quadrature (1st kind) :
 *
 * \f$ \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \simeq \sum_{i=1}^N \omega_i f(x_i)\f$
 *
 * where the Chebyshev nodes in [-1,1] are given by \f$ x_i = \cos(\frac{2i-1}{2N}\pi)\f$ and the
 * weights \f$ \omega_i = \pi/N \f$
 *
 * Here we will transpose this formula in the SDM context where, the Chebyshev
 * points are defined on interval [0,1].
 *
 * After the change of variables \f$ x' = (x+1)/2 \f$, one obtains
 * \f$ \int_{0}^{1} f(x) dx \simeq \sum_{i=1}^N \omega_i f(x_i) \sqrt{x_i(1-x_i)}\f$ where now the
 * quadrature points are \f$ x_i = 0.5 \( 1 - \cos (\frac{2i-1}{2N}\pi) \) \f$ as in the SDM
 * solution points.
 *
 * See also:
 * http://mathworld.wolfram.com/Chebyshev-GaussQuadrature.html
 * https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <math.h> // for M_PI

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

namespace ppkMHD
{

using f_t = real_t (*)(real_t);

// some polynomial functions
real_t
f_0(real_t x)
{
  return 2.23;
}
real_t
f_1(real_t x)
{
  return 2 * x + 1;
}
real_t
f_2(real_t x)
{
  return x * x + 1;
}
real_t
f_3(real_t x)
{
  return -x * x * x + 4 * x * x + x - 1;
}
real_t
f_4(real_t x)
{
  return 2 * x * x * x * x - 4 * x * x + 7 * x + 3;
}
real_t
f_5(real_t x)
{
  return 2.5 * x * x * x * x * x - 16 * x * x + x + 1;
}
real_t
f_6(real_t x)
{
  return 5 * x * x * x * x * x * x - 16 * x * x * x - x - 5;
}
real_t
f_7(real_t x)
{
  return cos(x) + 0.23 * 2 * x;
}
real_t
f_8(real_t x)
{
  return 1.0 / (x + 1);
}

// the corresponding primitives
real_t
pf_0(real_t x)
{
  return 2.23 * x;
}
real_t
pf_1(real_t x)
{
  return x * x + x;
}
real_t
pf_2(real_t x)
{
  return x * x * x / 3 + x;
}
real_t
pf_3(real_t x)
{
  return -x * x * x * x / 4 + 4 * x * x * x / 3 + x * x / 2 - x;
}
real_t
pf_4(real_t x)
{
  return 2 * x * x * x * x * x / 5 - 4 * x * x * x / 3 + 7 * x * x / 2 + 3 * x;
}
real_t
pf_5(real_t x)
{
  return 2.5 * x * x * x * x * x * x / 6 - 16 * x * x * x / 3 + x * x / 2 + 1;
}
real_t
pf_6(real_t x)
{
  return 5 * x * x * x * x * x * x * x / 7 - 16 * x * x * x * x / 4 - x * x / 2 - 5 * x;
}
real_t
pf_7(real_t x)
{
  return sin(x) + 0.23 * x * x;
}
real_t
pf_8(real_t x)
{
  return log(x + 1);
}

// select function for exact reconstruction
enum func_type_t
{
  FUNCTION,
  PRIMITIVE
};

/**
 *
 */
template <int func_type>
f_t
select_function(int i)
{

  if (func_type == FUNCTION)
  {

    if (i == 0)
      return f_0;
    else if (i == 1)
      return f_1;
    else if (i == 2)
      return f_2;
    else if (i == 3)
      return f_3;
    else if (i == 4)
      return f_4;
    else if (i == 5)
      return f_5;
    else if (i == 6)
      return f_6;
    else if (i == 7)
      return f_7;
    else if (i == 8)
      return f_8;

    // default
    return f_3;
  }
  else
  {

    if (i == 0)
      return pf_0;
    else if (i == 1)
      return pf_1;
    else if (i == 2)
      return pf_2;
    else if (i == 3)
      return pf_3;
    else if (i == 4)
      return pf_4;
    else if (i == 5)
      return pf_5;
    else if (i == 6)
      return pf_6;
    else if (i == 7)
      return pf_7;
    else if (i == 8)
      return pf_8;

    // default
    return pf_3;
  }
}

/*
 *
 * Main test.
 *
 * \param[in] i number of the function to test for integration
 *
 * \tparam dim is either 2 or 3 (don't really matter here, since we only use the
 *     1d points)
 * \tparam N   number of quadrature points
 *
 */
template <int dim, int N>
void
test_chebyshev_quadrature(int i)
{

  std::cout << "=========================================================\n";

  std::cout << "Using function number " << i << "\n";

  // function pointer setup
  // f is the function to integrate
  f_t f = select_function<FUNCTION>(i);

  // exact primitive (for error computation)
  f_t pf = select_function<PRIMITIVE>(i);

  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order  : " << N << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  std::cout << "  Number of flux     points : " << N + 1 << "\n";

  sdm::SDM_Geometry<dim, N> sdm_geom;

  sdm_geom.init(0);

  // std::cout << "Solution poins:\n";
  // for (int j=0; j<N; ++j) {
  //   for (int i=0; i<N; ++i) {
  //     std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
  // 		<< "," << sdm_geom.solution_pts_1d_host(j) << ") ";
  //   }
  //   std::cout << "\n";
  // }

  // integrale computed with quadrature
  real_t I_quad = 0;

  // exact value
  real_t I_exact;

  // weight
  real_t w = M_PI / N;

  // eval integral value
  // take care we made the change of variable x -> x'=(x+1)/2
  // so that x' is in [0,1] instead of [-1,1]
  for (int i = 0; i < N; ++i)
  {
    real_t x = sdm_geom.solution_pts_1d_host(i);
    I_quad += f(x) * sqrt(x - x * x);
  }
  I_quad *= w;

  I_exact = pf(1.0) - pf(0.0);

  printf("Integrale of f_%d on [0,1] is %f, exact value is %f, error is %f\n",
         i,
         I_quad,
         I_exact,
         I_quad - I_exact);

} // test_chebyshev_quadrature
} // namespace ppkMHD

// ==========================================================================
// ==========================================================================
// ==========================================================================
int
main(int argc, char * argv[])
{

  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }

    Kokkos::print_configuration(msg);

    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  std::cout << "=========================================================\n";
  std::cout << "=== Gauss-Chebyshev quadrature on interval [0,1] test ===\n";
  std::cout << "=========================================================\n";

  // testing for multiple value of N
  {
    // dim = 2
    ppkMHD::test_chebyshev_quadrature<2, 2>(3);
    ppkMHD::test_chebyshev_quadrature<2, 3>(4);
    ppkMHD::test_chebyshev_quadrature<2, 4>(6);
    ppkMHD::test_chebyshev_quadrature<2, 5>(6);
    ppkMHD::test_chebyshev_quadrature<2, 50>(6);
    ppkMHD::test_chebyshev_quadrature<2, 360>(7);
    ppkMHD::test_chebyshev_quadrature<2, 36>(8);
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // main
