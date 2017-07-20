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

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int N>
void test_sdm_geom_2d()
{

  std::cout << "===========\n";
  std::cout << "=====2D====\n";
  std::cout << "===========\n";
  
  // dim is the number of variable in the multivariate polynomial representation
  //int dim=2;
  
  sdm::SDM_Geometry<2,N> sdm_geom;
  
  sdm_geom.init(N);
  
  for (int j=0; j<N; ++j) {
    for (int i=0; i<N; ++i) {
      std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
		<< "," << sdm_geom.solution_pts_1d_host(j) << ") ";
    }
    std::cout << "\n";
  }    
  std::cout << "\n";

} // test_sdm_geom_2d<N>

template<int N>
void test_sdm_geom_3d()
{

  std::cout << "===========\n";
  std::cout << "=====3D====\n";
  std::cout << "===========\n";
  
  // dim is the number of variable in the multivariate polynomial representation
  //int dim=3;
  
  sdm::SDM_Geometry<3,N> sdm_geom;
  
  sdm_geom.init(N);
  
  for (int k=0; k<N; ++k) {
    for (int j=0; j<N; ++j) {
      for (int i=0; i<N; ++i) {
	std::cout << "(" << sdm_geom.solution_pts_1d_host(i)
		  << "," << sdm_geom.solution_pts_1d_host(j)
		  << "," << sdm_geom.solution_pts_1d_host(k) << ") ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n";
  }
  
} // test_sdm_geom_3d<N>

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


  // instantiate some tests
  test_sdm_geom_2d<2>();
  test_sdm_geom_2d<3>();
  test_sdm_geom_2d<4>();

  test_sdm_geom_3d<2>();
  test_sdm_geom_3d<3>();
  test_sdm_geom_3d<4>();

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
