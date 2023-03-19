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

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

namespace ppkMHD {

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
  UNUSED(argc);
  UNUSED(argv);

  int myRank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif // USE_MPI

  if (myRank==0) {
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
  }

  // read input file
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = dim == 2 ? "test_sdm_io_2D.ini" : "test_sdm_io_3D.ini";
  ConfigMap configMap(input_file);

  // create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // create solver
  sdm::SolverHydroSDM<dim,N> solver(params, configMap);

  // initialize the IO_ReadWrite object (normally done in
  // SolverFactory's create method)
  solver.init_io();

  solver.save_solution();

} // test_sdm_io

} // namespace ppkMHD

// =======================================================================
// =======================================================================
// =======================================================================
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

  {
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
      Kokkos::print_configuration( msg );
      std::cout << msg.str();
      std::cout << "##########################\n";
    }

    if (myRank==0) {
      std::cout << "===============================================\n";
      std::cout << "==== Spectral Difference Lagrange IO  test ====\n";
      std::cout << "===============================================\n";
    }


    // testing for multiple values of N in 2 to 6
    {

      // 2d
      ppkMHD::test_sdm_io<2,4>(argc,argv);

      // 3d
      ppkMHD::test_sdm_io<3,4>(argc,argv);

    }
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // main
