/**
 * This executable is used to test sdm::SDM_Geometry class.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

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
  int order = 4;
  
  if (argc>1)
    order = atoi(argv[2]);

  // ===========
  // 2D test
  // ===========

  {

    std::cout << "===========\n";
    std::cout << "=====2D====\n";
    std::cout << "===========\n";
    
    // dim is the number of variable in the multivariate polynomial representation
    int dim=2;

    sdm::SDM_Geometry<2> sdm_geom;

    sdm_geom.init(order);

    for (int j=0; j<order; ++j) {
      for (int i=0; i<order; ++i) {
	std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
		  << "," << sdm_geom.solution_pts_1d_host(j) << ") ";
      }
      std::cout << "\n";
    }

  }    

  // ===========
  // 3D test
  // ===========

  {
    std::cout << "===========\n";
    std::cout << "=====3D====\n";
    std::cout << "===========\n";

    // dim is the number of variable in the multivariate polynomial representation
    int dim=3;

    sdm::SDM_Geometry<3> sdm_geom;

    sdm_geom.init(order);

    for (int k=0; k<order; ++k) {
      for (int j=0; j<order; ++j) {
	for (int i=0; i<order; ++i) {
	  std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
		    << "," << sdm_geom.solution_pts_1d_host(j)
		    << "," << sdm_geom.solution_pts_1d_host(k) << ") ";
	}
	std::cout << "\n";
      }
	std::cout << "\n\n";
    }

  }    

  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
