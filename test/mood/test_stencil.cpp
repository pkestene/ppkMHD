/**
 * This executable is used to test Stencil class.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "mood/Stencil.h"
#include "shared/real_type.h"

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
  unsigned int dim=3;

  // highest degree / order of the polynomial
  int order = 4;
  
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

  mood::Stencil stencil(stencilId);

  stencil.print_stencil();
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
