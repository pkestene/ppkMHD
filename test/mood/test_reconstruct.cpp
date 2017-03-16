/**
 * This executable is used to test polynomial reconstruction.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"

// mood
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

/**
 * a simple polynomial test function.
 */
double test_function_2d(double x, double y)
{

  return x*x+2.2*x*y+4.1*y*y-5.0+x;
  
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

  // dim is the number of variable in the multivariate polynomial representation
  unsigned int dim=2;

  // highest degree / order of the polynomial
  int order = 2;
  
  if (argc>1)
    dim = atoi(argv[1]);
  if (argc>2)
    order = atoi(argv[2]);

  mood::STENCIL_ID stencilId = mood::Stencil::select_stencil(dim,order);

  /*
   * test class Polynomial.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Stencil    \n";

  std::cout << "############################\n";

  mood::Stencil stencil = mood::Stencil(stencilId);

  mood::StencilUtils::print_stencil(stencil);

  real_t dx, dy, dz;
  dx = dy = dz = 0.1;
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int stencil_size   = mood::get_stencil_size(stencilId);
  int stencil_degree = mood::get_stencil_degree(stencilId);
  mood::Matrix geomMatrix(stencil_size,stencil_degree);
    
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
