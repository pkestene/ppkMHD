/**
 * This executable is used to test sdm::SDM_Geometry class, 
 * more specificly the computation of a derivative using Lagrange 
 * polynomial representation.
 *
 * Please be aware the following test is OK as long as we are using polynomial
 * representation in interval [0,1]. If you use something else, you wil have
 * to "rescale" the derivative. 
 * 
 * \author Pierre Kestener
 * \date Aug 24, 2017
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"


//! 4th order polynomial
real_t f_4(real_t x)
{
  return  2*x*x*x*x - 4*x*x + 7*x + 3;
}

//! derivative of the previous polynomial
real_t df_4(real_t x)
{
  return 8*x*x*x - 8*x + 7;
}

using f_t = real_t (*)(real_t);


/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int dim,
	 int N>
void test_lagrange_derivative()
{

  std::cout << "=========================================================\n";

  // function pointer setup for interpolation values
  // remember that with N solution points, one can recontruct exactly
  // polynomials up to degree N-1; so here we test the exact reconstruction.

  // example function and its exact derivative
  f_t  f = f_4;
  f_t df = df_4;
  
  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order  : " << N << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  
  sdm::SDM_Geometry<dim,N> sdm_geom;

  sdm_geom.init(0);
    
  sdm_geom.init_lagrange_1d();
  
  std::cout << "1D lagrange derivative evaluation (at solution points):\n";

  // some useful types to hold values at solution points
  using DataVal     = Kokkos::View<real_t*,Device>;
  using DataValHost = Kokkos::View<real_t*,Device>::HostMirror;

  DataVal     solution_values   = DataVal("solution_values",N);
  DataValHost solution_values_h = Kokkos::create_mirror(solution_values);
  
  // create values at solution points:
  for (int i=0; i<N; ++i)
    solution_values_h(i) = f(sdm_geom.solution_pts_1d_host(i));

  // some useful types (the same as in SDM_Geometry class)
  using LagrangeMatrix     = Kokkos::View<real_t **, Device>;
  using LagrangeMatrixHost = LagrangeMatrix::HostMirror;

  // retrieve on host the Lagrange polynomial derivative matrix
  LagrangeMatrixHost sol2sol_derivative_h = Kokkos::create_mirror(sdm_geom.sol2sol_derivative);
  Kokkos::deep_copy(sol2sol_derivative_h, sdm_geom.sol2sol_derivative);

  // print matrix
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      std::cout << sol2sol_derivative_h(i,j) << " ";
    }
    std::cout << "\n";
  }
  
  // evaluate derivative at solution points
  for (int j=0; j<N; ++j) {
    
    // compute derivative value
    // remember that sol2sol_derivative_h(k,j) is the derivative of the k-th
    // Lagrange polynomial evaluated at the j-th solution points
    real_t val=0;
    for (int k=0; k<N; ++k) {
      val += solution_values_h(k) * sol2sol_derivative_h(k,j);
    }
    
    real_t x_j = sdm_geom.solution_pts_1d_host(j);
    
    printf("Evaluated derivative of function f at %f is %f - exact value is %f (difference with exact value is %e)\n",x_j,val, df(x_j), val-df(x_j));
    
  }
  
} // test_lagrange_derivative

/*************************************************/
/*************************************************/
/*************************************************/
int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "      KOKKOS CONFIG       \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
#if defined( CUDA )
    Kokkos::Cuda::print_configuration( msg );
#else
    Kokkos::OpenMP::print_configuration( msg );
#endif
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  std::cout << "=========================================================\n";
  std::cout << "==== Spectral Difference Lagrange Interpolation test ====\n";
  std::cout << "=========================================================\n";

  // testing for multiple value of N in 2 to 6
  {
    // 2d
    test_lagrange_derivative<2,2>();
    test_lagrange_derivative<2,3>();
    test_lagrange_derivative<2,4>();
    test_lagrange_derivative<2,5>();
    test_lagrange_derivative<2,6>();

    // 3d
    //test_lagrange<3,2>();
    //test_lagrange<3,3>();
    //test_lagrange<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
