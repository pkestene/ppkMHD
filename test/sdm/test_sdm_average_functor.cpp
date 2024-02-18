/**
 * This executable is used to test SDM average HydroState computation functor.
 *
 * Take care the parameter file must enable limiter computations, i.e. it must
 * contains a INI section:
 * ...
 * [sdm]
 * limiter_enabled=true
 * ...
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <memory> // for shared_ptr

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/SolverHydroSDM.h"
#include "sdm/HydroInitFunctors.h"
#include "sdm/SDM_Limiter_Functors.h"

// for IO
#include "utils/io/IO_ReadWrite_SDM.h"
#include "utils/io/IO_ReadWrite.h"

#ifdef USE_MPI
#  include "utils/mpiUtils/GlobalMpiSession.h"
#  include <mpi.h>
#endif // USE_MPI

namespace ppkMHD
{

/**
 * Wrapper routine arround the functor call
 * sdm::Average_Conservative_Variables_Functor
 */
template <int dim, int N>
void
compute_Uaverage(sdm::SolverHydroSDM<dim, N> & solver)
{

  int nbCells = dim == 2 ? solver.params.isize * solver.params.jsize
                         : solver.params.isize * solver.params.jsize * solver.params.ksize;

  // compute cell average
  {
    sdm::Average_Conservative_Variables_Functor<dim, N> functor(
      solver.params, solver.sdm_geom, solver.U, solver.Uaverage);

    Kokkos::parallel_for(nbCells, functor);
  }

  // compute x gradient cell-averaged
  {
    sdm::Average_Gradient_Functor<dim, N, IX> functor(
      solver.params, solver.sdm_geom, solver.U, solver.Ugradx);
    Kokkos::parallel_for(nbCells, functor);
  }

  // compute y gradient cell-averaged
  {
    sdm::Average_Gradient_Functor<dim, N, IY> functor(
      solver.params, solver.sdm_geom, solver.U, solver.Ugrady);
    Kokkos::parallel_for(nbCells, functor);
  }

  // compute z gradient cell-averaged
  if (dim == 3)
  {
    sdm::Average_Gradient_Functor<dim, N, IZ> functor(
      solver.params, solver.sdm_geom, solver.U, solver.Ugradz);
    Kokkos::parallel_for(nbCells, functor);
  }

  return;

} // compute_Uaverage

/**
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template <int dim, int N>
void
test_compute_average_functor()
{

  int myRank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif // USE_MPI

  if (myRank == 0)
  {
    std::cout << "===============================================\n";
    std::cout << "===============================================\n";
    std::cout << "===============================================\n";
    std::cout << "  Dimension is : " << dim << "\n";
    std::cout << "  Using order : " << N << "\n";
    std::cout << "  Number of solution points : " << N << "\n";
    std::cout << "  Number of flux     points : " << N + 1 << "\n";
    std::cout << "===============================================\n";
    std::cout << "===============================================\n";
    std::cout << "===============================================\n";
  }

  // read input file
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = dim == 2 ? "test_sdm_limiter_2D.ini" : "test_sdm_limiter_3D.ini";
  ConfigMap   configMap(input_file);

  // create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // create solver
  sdm::SolverHydroSDM<dim, N> solver(params, configMap);

  // initialize the IO_ReadWrite object (normally done in
  // SolverFactory's create method)
  solver.init_io();

  // save data just for cross-checking
  solver.save_solution();

  // actual test here
  if (solver.limiter_enabled)
    compute_Uaverage<dim, N>(solver);

  // save average data
  std::shared_ptr<ppkMHD::io::IO_ReadWrite> io_writer_average =
    std::make_shared<ppkMHD::io::IO_ReadWrite>(params, configMap, solver.m_variables_names);

  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  {
    typename DataArray::HostMirror data_host = Kokkos::create_mirror(solver.Uaverage);
    io_writer_average->save_data(solver.Uaverage, data_host, 0, 0.0, "");
  }

  {
    typename DataArray::HostMirror data_host = Kokkos::create_mirror(solver.Ugradx);
    io_writer_average->save_data(solver.Ugradx, data_host, 0, 0.0, "gradx");
  }

  {
    typename DataArray::HostMirror data_host = Kokkos::create_mirror(solver.Ugrady);
    io_writer_average->save_data(solver.Ugrady, data_host, 0, 0.0, "grady");
  }

  if (dim == 3)
  {
    typename DataArray::HostMirror data_host = Kokkos::create_mirror(solver.Ugradz);
    io_writer_average->save_data(solver.Ugradz, data_host, 0, 0.0, "gradz");
  }


} // test_compute_dt_functors

} // namespace ppkMHD

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
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  std::cout << "==========================================================\n";
  std::cout << "=== Spectral Difference Method : Limiter functors test ===\n";
  std::cout << "==========================================================\n";

  // testing for multiple value of N in 2 to 6
  {
    // 2d
    ppkMHD::test_compute_average_functor<2,4>();

    // 3d
    ppkMHD::test_compute_average_functor<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

}
