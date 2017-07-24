/**
 * Testing ideas for least-squares estimation of gradient at cell center,
 * using data at solution nodes.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int dim,
	 int N>
void test_sdm_lsq_gradient()
{

  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order : "  << N   << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  std::cout << "  Number of flux     points : " << N+1 << "\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  

  
} // test_sdm_lsq_gradient

// =====================================================================
// =====================================================================
// =====================================================================
int main(int argc, char* argv[])
{

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI

  int myRank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif // USE_MPI

  
  Kokkos::initialize(argc, argv);

  if (myRank==0) {
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

  if (myRank==0) {
    std::cout << "================================================\n";
    std::cout << "==== Spectral Difference Lagrange      test ====\n";
    std::cout << "==== Least-square gradient estimation ==========\n";
    std::cout << "================================================\n";
  }
  
  
  // testing for multiple values of N between 2 to 6 in 2d and 3d
  {

    // 2d
    test_sdm_lsq_gradient<2,4>();

    // 3d
    test_sdm_lsq_gradient<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}

