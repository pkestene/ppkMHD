/**
 * This executable is used to test sdm::SolverHydroSDM class, 
 * more specificly init conditions and vtk output.
 *
 * For output, we would like to output multiple values per cell, in order to
 * "visualize" the high-order quality of the SDM scheme.
 *
 * About what others do:
 * - Deal.ii uses a class named DataOut, which has a method build_patches
 *   build_patches (const unsigned int n_subdivisions=0)
 *   which allows when outputing data coming from a Discontinuous Galerkin
 *   scheme to subdivide each cell, a recompute some local interpolation on a 
 *   refine patche.
 *   see https://www.dealii.org/8.5.0/doxygen/deal.II/classDataOut.html
 * - see also MFEM : https://github.com/mfem
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/SolverHydroSDM.h"

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

//! 5th order polynomial
real_t f_5(real_t x)
{
  return 2.5*x*x*x*x*x - 16*x*x + x + 1;
}

//! 6th order polynomial
real_t f_6(real_t x)
{
  return 5*x*x*x*x*x*x - 16*x*x*x - x - 5;
}

using f_t = real_t (*)(real_t);

// select polynomial for exact reconstruction
f_t select_polynomial(int N) {

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

// select polynomial for non-exact reconstruction
f_t select_polynomial_non_exact(int N) {

  if (N==1)
    return f_1;
  else if (N==2)
    return f_2;
  else if (N==3)
    return f_3;
  else if (N==4)
    return f_4;
  else if (N==5)
    return f_5;
  else if (N==6)
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
template<int dim,
	 int N>
void test_sdm_io(int argc, char* argv[])
{

  std::cout << "===============================================\n";

  // function pointer setup for interpolation values
  // remember that with N solution points, one can recontruct exactly
  // polynomials up to degree N-1; so here we test the exact reconstruction.
  f_t f = select_polynomial(N);
  //f_t f = select_polynomial_non_exact(N);
  
  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order : "  << N   << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  std::cout << "  Number of flux     points : " << N+1 << "\n";
  
  sdm::SDM_Geometry<dim,N> sdm_geom;

  // read input file
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);

  // create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // create solver
  sdm::SolverHydroSDM<dim,N> solver(params, configMap);

  solver.save_solution();
  
} // test_sdm_io

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

  std::cout << "===============================================\n";
  std::cout << "==== Spectral Difference Lagrange IO  test ====\n";
  std::cout << "===============================================\n";

  if (argc<2) {
    std::cout << "Please provide a settings file\n";
    std::cout << "Usage : ./test_sdm_io test.ini\n";
    return EXIT_FAILURE;
  }
    
  
  // testing for multiple values of N in 2 to 6
  {
    // 2d
    //test_sdm_io<2,4>(argc,argv);

    // 3d
    test_sdm_io<3,4>(argc,argv);

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
