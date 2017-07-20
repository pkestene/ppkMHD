/**
 * This executable is used to test SDM CFL constraint computation functor.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/SolverHydroSDM.h"
#include "sdm/HydroInitFunctors.h"
#include "sdm/SDM_Dt_Functor.h"

// for IO
#include "utils/io/IO_Writer_SDM.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

template< int dim, int N >
real_t compute_dt(sdm::SolverHydroSDM<dim,N>& solver) {
    
  // create equation system object
  ppkMHD::EulerEquations<dim> euler;
  
  int nbCells = dim==2 ?
    solver.params.isize * solver.params.jsize :
    solver.params.isize * solver.params.jsize * solver.params.ksize;
  
  real_t invDt = 0.0;

  // alias to the computational functor, dimension dependend
  using ComputeDtFunctor =
    typename std::conditional<dim==2,
                              sdm::ComputeDt_Functor_2d<N>,
                              sdm::ComputeDt_Functor_3d<N> >::type;
  ComputeDtFunctor functor(solver.params,
			   solver.sdm_geom,
			   euler,
			   solver.U);
  
  Kokkos::parallel_reduce(nbCells, functor, invDt);
  
  real_t dt = solver.params.settings.cfl/invDt;
  printf("dt = %f (invDt = %f)\n", dt,invDt);
  
  return dt;
  
} // compute_dt

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int dim,
	 int N>
void test_compute_dt_functors()
{

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

  // initialize the IO_Writer object (normally done in
  // SolverFactory's create method)
  solver.init_io_writer();
  
  // save data just for cross-checking
  solver.save_solution();

  // actual test here
  compute_dt<dim,N>(solver);

} // test_compute_dt_functors

/*************************************************/
/*************************************************/
/*************************************************/
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

  std::cout << "=========================================================\n";
  std::cout << "==== Spectral Difference Method : CFL functors test ====\n";
  std::cout << "=========================================================\n";

  // testing for multiple value of N in 2 to 6
  {
    // 2d
    test_compute_dt_functors<2,4>();

    // 3d
    test_compute_dt_functors<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
