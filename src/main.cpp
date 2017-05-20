/**
 * Hydro/MHD solver (Muscl-Hancock).
 *
 * \date April, 16 2016
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>

#include "shared/kokkos_shared.h"

#include "shared/real_type.h"   // choose between single and double precision
#include "shared/HydroParams.h" // read parameter file

// solver
#include "shared/SolverFactory.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
//#include "shared/HydroParamsMpi.h" // read parameter file
#include <mpi.h>
#endif // USE_MPI

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char *argv[])
{

  using namespace ppkMHD;

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI
  
#ifdef CUDA
  // Initialize Host mirror device
  Kokkos::HostSpace::execution_space::initialize(1);
  const unsigned device_count = Kokkos::Cuda::detect_device_count();

  // Use the last device:
  Kokkos::Cuda::initialize( Kokkos::Cuda::SelectDevice(device_count-1) );
#else
  Kokkos::initialize(argc, argv);
#endif

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

#ifdef USE_MPI
# ifdef CUDA
    {
      int rank, nRanks;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
      
      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
		<< " pinned to GPU #" << cudaDeviceId << "\n";
    }
# endif // CUDA
#endif // USE_MPI

    
  }

  if (argc != 2) {
    fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
    exit(EXIT_FAILURE);
  }

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  SolverBase *solver = SolverFactory::Instance().create(solver_name,
							params,
							configMap);

  if (params.nOutput != 0)
    solver->save_solution();
  
  // start computation
  std::cout << "Start computation....\n";
  solver->timers[TIMER_TOTAL]->start();

  // Hydrodynamics solver loop
  while ( ! solver->finished() ) {

    solver->next_iteration();

  } // end solver loop

  // end of computation
  solver->timers[TIMER_TOTAL]->stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();

  printf("final time is %f\n", solver->m_t);
  
  // print monitoring information
  {    
    real_t t_tot   = solver->timers[TIMER_TOTAL]->elapsed();
    real_t t_comp  = solver->timers[TIMER_NUM_SCHEME]->elapsed();
    real_t t_dt    = solver->timers[TIMER_DT]->elapsed();
    real_t t_bound = solver->timers[TIMER_BOUNDARIES]->elapsed();
    real_t t_io    = solver->timers[TIMER_IO]->elapsed();
    printf("total       time : %5.3f secondes\n",t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n",t_comp,100*t_comp/t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n",t_dt,100*t_dt/t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n",t_bound,100*t_bound/t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n",t_io,100*t_io/t_tot);
    printf("Perf             : %10.2f number of Mcell-updates/s\n",solver->m_iteration*solver->m_nCells/t_tot*1e-6);
  }

  delete solver;

#ifdef CUDA
  Kokkos::Cuda::finalize();
  Kokkos::HostSpace::execution_space::finalize();
#else
  Kokkos::finalize();
#endif
  
  return EXIT_SUCCESS;

} // end main
