/**
 * This executable is used to test sdm::SDM_Geometry class, 
 * more specific Lagrange interpolation.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

//! const polynomial
real_t f_0(real_t x)
{
  return 2.23;
}

//! 1st order polynomial
real_t f_1(real_t x)
{
  return 2 * x + 1;
}

//! 2nd order polynomial
real_t f_2(real_t x)
{
  return x * x + 1;
}

//! 3rd order polynomial
real_t f_3(real_t x)
{
  return - x*x*x + 4*x*x + x - 1;
}

//! 4th order polynomial
real_t f_4(real_t x)
{
  return  2*x*x*x*x - 4*x*x + 7*x + 3;
}

//! 5rd order polynomial
real_t f_5(real_t x)
{
  return 2.5*x*x*x*x*x - 16*x*x + x + 1;
}

using f_t = real_t (*)(real_t);

f_t select_polynomial(int& N) {

  if (N==1)
    return f_0;
  else if (N==2)
    return f_1;
  else if (N==3)
    return f_2;
  else if (N==4)
    return f_3;
  else if (N==5)
    return f_4;
  else if (N==6)
    return f_5;

  // default
  return f_3;
  
}


int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
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

  // highest degree / order of the polynomial
  int N = 2;
  
  if (argc>1)
    N = atoi(argv[1]);

  // check
  if (N<1 or N>6)
    N=3;
  
  // function pointer setup for interpolation values
  // remember that with N solution points, one can recontruct exactly
  // polynomials up to degree N-1; so here we test the exact reconstruction.
  f_t f = select_polynomial(N);
  
  {

    std::cout << "=========================================================\n";
    std::cout << "==== Spectral Difference Lagrange Interpolation test ====\n";
    std::cout << "=========================================================\n";

    std::cout << "  Using order : " << N << "\n";
    std::cout << "  Number of solution points : " << N << "\n";
    std::cout << "  Number of flux     points : " << N+1 << "\n";
    
    // dim is the number of variable in the multivariate polynomial representation
    int dim=2;

    sdm::SDM_Geometry<2> sdm_geom;

    sdm_geom.init(N);

    std::cout << "Solution poins:\n";
    for (int j=0; j<N; ++j) {
      for (int i=0; i<N; ++i) {
	std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
		  << "," << sdm_geom.solution_pts_1d_host(j) << ") ";
      }
      std::cout << "\n";
    }

    sdm_geom.init_lagrange_1d();

    std::cout << "1D lagrange interpolation solution to flux:\n";

    // create values at solution points:
    using DataVal = Kokkos::View<real_t*>;
    DataVal solution_values = DataVal("solution_values",N);
    DataVal solution_values_h = Kokkos::create_mirror(solution_values);

    for (int i=0; i<N; ++i)
      solution_values_h(i) = f(sdm_geom.solution_pts_1d_host(i));

    for (int j=0; j<N+1; ++j) {

      // compute interpolated value
      real_t val=0;
      for (int k=0; k<N; ++k) {
	val += solution_values_h(k) * sdm_geom.sol2flux(k,j);
      }

      real_t x_j = sdm_geom.flux_pts_1d_host(j);
      
      printf("Interpolated value at %f is %f as compared to exact value %f\n",x_j,val, f(x_j));
      
    }
    
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
