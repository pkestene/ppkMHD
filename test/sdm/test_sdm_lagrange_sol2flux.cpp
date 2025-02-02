/**
 * This executable is used to test sdm::SDM_Geometry class,
 * more specific Lagrange interpolation.
 *
 * More precicely, given a Lagrange polynomial basis at solution points, we
 * interpolate values at flux points.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

namespace ppkMHD
{

//! const polynomial
real_t
f_0(real_t x)
{
  return 2.23;
}

//! 1st order polynomial
real_t
f_1(real_t x)
{
  return 2 * x + 1;
}

//! 2nd order polynomial
real_t
f_2(real_t x)
{
  return x * x + 1;
}

//! 3rd order polynomial
real_t
f_3(real_t x)
{
  return -x * x * x + 4 * x * x + x - 1;
}

//! 4th order polynomial
real_t
f_4(real_t x)
{
  return 2 * x * x * x * x - 4 * x * x + 7 * x + 3;
}

//! 5th order polynomial
real_t
f_5(real_t x)
{
  return 2.5 * x * x * x * x * x - 16 * x * x + x + 1;
}

//! 6th order polynomial
real_t
f_6(real_t x)
{
  return 5 * x * x * x * x * x * x - 16 * x * x * x - x - 5;
}

using f_t = real_t (*)(real_t);

// select polynomial for exact reconstruction
f_t
select_polynomial(int N)
{

  if (N == 1)
    return f_0;
  else if (N == 2)
    return f_1;
  else if (N == 3)
    return f_2;
  else if (N == 4)
    return f_3;
  else if (N == 5)
    return f_4;
  else if (N == 6)
    return f_5;

  // default
  return f_3;
}

// select polynomial for non-exact reconstruction
f_t
select_polynomial_non_exact(int N)
{

  if (N == 1)
    return f_1;
  else if (N == 2)
    return f_2;
  else if (N == 3)
    return f_3;
  else if (N == 4)
    return f_4;
  else if (N == 5)
    return f_5;
  else if (N == 6)
    return f_6;

  // default
  return f_3;
}

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template <int dim, int N>
void
test_lagrange()
{

  std::cout << "=========================================================\n";

  // function pointer setup for interpolation values
  // remember that with N solution points, one can reconstruct exactly
  // polynomials up to degree N-1; so here we test the exact reconstruction.
  f_t f = select_polynomial(N);
  // f_t f = select_polynomial_non_exact(N);

  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order  : " << N << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  std::cout << "  Number of flux     points : " << N + 1 << "\n";

  sdm::SDM_Geometry<dim, N> sdm_geom;

  sdm_geom.init(0);

  std::cout << "Solution points:\n";
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      std::cout << "(" << sdm_geom.solution_pts_1d_host(i) << ","
                << sdm_geom.solution_pts_1d_host(j) << ") ";
    }
    std::cout << "\n";
  }

  sdm_geom.init_lagrange_1d();

  std::cout << "1D lagrange interpolation solution to flux:\n";

  // create values at solution points:
  using DataVal = Kokkos::View<real_t *, Device>;
  using DataValHost = Kokkos::View<real_t *, Device>::HostMirror;
  DataVal     solution_values = DataVal("solution_values", N);
  DataValHost solution_values_h = Kokkos::create_mirror(solution_values);

  for (int i = 0; i < N; ++i)
    solution_values_h(i) = f(sdm_geom.solution_pts_1d_host(i));

  using LagrangeMatrix = Kokkos::View<real_t **, Device>;
  using LagrangeMatrixHost = LagrangeMatrix::HostMirror;

  LagrangeMatrixHost sol2flux_h = Kokkos::create_mirror(sdm_geom.sol2flux);
  Kokkos::deep_copy(sol2flux_h, sdm_geom.sol2flux);

  for (int j = 0; j < N + 1; ++j)
  {

    // compute interpolated value
    real_t val = 0;
    for (int k = 0; k < N; ++k)
    {
      val += solution_values_h(k) * sol2flux_h(k, j);
    }

    real_t x_j = sdm_geom.flux_pts_1d_host(j);

    printf("Interpolated value at %f is %f as compared to exact value %f\n", x_j, val, f(x_j));
  }

} // test_lagrange

} // namespace ppkMHD

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

    std::cout << "=========================================================\n";
    std::cout << "==== Spectral Difference Lagrange Interpolation test ====\n";
    std::cout << "=========================================================\n";

    // testing for multiple value of N in 2 to 6
    {
      // 2d
      ppkMHD::test_lagrange<2, 2>();
      ppkMHD::test_lagrange<2, 3>();
      ppkMHD::test_lagrange<2, 4>();
      ppkMHD::test_lagrange<2, 5>();
      ppkMHD::test_lagrange<2, 6>();

      // 3d
      ppkMHD::test_lagrange<3, 2>();
      ppkMHD::test_lagrange<3, 3>();
      ppkMHD::test_lagrange<3, 4>();
    }
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // main
