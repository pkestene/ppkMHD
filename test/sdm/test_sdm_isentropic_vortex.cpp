/**
 * Test high-order SDM scheme convergence using the isentropic vortex test.
 *
 * Perform simulation from t=0 to t=10.0 and compute L1 / L2 error.
 *
 * \date July, 21st 2017
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>

#include "shared/kokkos_shared.h"

#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information

// solver
//#include "shared/SolverFactory.h"
#include "sdm/SolverHydroSDM.h"

// compare / compute L1/L2 norm of the difference between solver solution
// and the exact solution (which is also the initial condition)
#include "sdm/SDM_Compute_error.h"

enum RK_type {
  FORWARD_EULER=1,
  SSP_RK2=2,
  SSP_RK3=3
};

/**
 * Generate a test parameter file.
 *
 * \param[in] N order of the SDM scheme; allowed values in 1..6
 * \param[in] size number of cells per dimension
 * \param[in] runge_kutta : 1 -> forward_euler, 2 -> ssprk2, 3 -> ssprk3
 */
void generate_input_file(int N, int size, int runge_kutta)
{

  std::fstream outFile;
  outFile.open ("test_sdm_isentropic_vortex.ini", std::ios_base::out);
  
  outFile << "[run]\n";
  outFile << "solver_name=Hydro_SDM_2D_degree" << N << "\n";
  outFile << "tEnd=10\n";
  outFile << "nStepmax=1000000\n";
  outFile << "nOutput=1\n";
  outFile << "\n";

  outFile << "[mesh]\n";
  outFile << "nx=" << size << "\n";
  outFile << "ny=" << size << "\n";
  outFile << "\n";

  outFile << "xmin=-5.0\n";
  outFile << "xmax=5.0\n";
  outFile << "\n";
  
  outFile << "ymin=-5.0\n";
  outFile << "ymax=5.0\n";
  outFile << "\n";

  outFile << "boundary_type_xmin=3\n";
  outFile << "boundary_type_xmax=3\n";
  outFile << "\n";

  outFile << "boundary_type_ymin=3\n";
  outFile << "boundary_type_ymax=3\n";
  outFile << "\n";
  
  outFile << "[hydro]\n";
  outFile << "gamma0=1.666\n";
  outFile << "cfl=0.8\n";
  outFile << "niter_riemann=10\n";
  outFile << "iorder=2\n";
  outFile << "slope_type=2\n";
  outFile << "problem=isentropic_vortex\n";
  outFile << "riemann=hllc\n";
  outFile << "\n";
  
  outFile << "[sdm]\n";
  if (runge_kutta == FORWARD_EULER) {
    outFile << "forward_euler=yes\n";
    outFile << "ssprk2=no\n";
    outFile << "ssprk3=no\n";
  } else if (runge_kutta == SSP_RK2) {
    outFile << "forward_euler=no\n";
    outFile << "ssprk2=yes\n";
    outFile << "ssprk3=no\n";
  } else if (runge_kutta == SSP_RK3) {
    outFile << "forward_euler=no\n";
    outFile << "ssprk2=no\n";
    outFile << "ssprk3=yes\n";
  }
  outFile << "\n";
  
  outFile << "[isentropic_vortex]\n";
  outFile << "density_ambient=1.0\n";
  outFile << "temperature_ambient=1.0\n";
  outFile << "vx_ambient=1.0\n";
  outFile << "vy_ambient=1.0\n";
  outFile << "vz_ambient=1.0\n";
  outFile << "\n";
  
  outFile << "[output]\n";
  outFile << "outputDir=./\n";
  outFile << "outputPrefix=test_isentropic_vortex_2D\n";
  outFile << "outputVtkAscii=false\n";
  outFile << "\n";
  
  outFile << "[other]\n";
  outFile << "implementationVersion=0\n";
  outFile << "\n";

  outFile.close();

} // generate_input_file

// ===============================================================
// ===============================================================
// ===============================================================
template<int N, int norm_type>
real_t compute_L2_versus_exact(sdm::SolverHydroSDM<2,N>* solver)
{
  
  real_t norm = 0.0;
  
  int nbCells =
    solver->params.isize *
    solver->params.jsize;
  
  // retrieve exact solution in auxiliary data arrary : solver.Uaux
  {
    IsentropicVortexParams iparams(solver->configMap);

    sdm::InitIsentropicVortexFunctor<2,N> functor(solver->params,
						  solver->sdm_geom,
						  iparams,
						  solver->Uaux);
    Kokkos::parallel_for(nbCells, functor);
  }

  // perform the actual comparison
  {
    sdm::Compute_Error_Functor_2d<N,norm_type> functor(solver->params,
						       solver->sdm_geom,
						       solver->U,
						       solver->Uaux,
						       ID);
    Kokkos::parallel_reduce(nbCells, functor, norm);
  }
  
  return norm/nbCells/N/N; 
  
} // compute_L2_versus_exact

// ===============================================================
// ===============================================================
// ===============================================================
template<int N, int norm_type>
real_t test_isentropic_vortex(int size, int runge_kutta)
{

  using namespace ppkMHD;

  std::cout << "###############################\n";
  std::cout << "Running isentropic vortex test \n";
  std::cout << "N    = " << N << "\n";
  std::cout << "size = " << size << "\n";
  std::cout << "Runge-Kutta order= " << runge_kutta << "\n";
  std::cout << "###############################\n";

  generate_input_file(N,size,runge_kutta);
  
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string("test_sdm_isentropic_vortex.ini");
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  sdm::SolverHydroSDM<2,N>* solver = new sdm::SolverHydroSDM<2,N>(params, configMap);
  solver->init_io_writer();
  
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

  real_t error = compute_L2_versus_exact<N,norm_type>(solver);
  
  print_solver_monitoring_info(solver);

  printf("test isentropic vortex for N=%d, size=%d, error=%5.3f\n",N,size,error);
  
  delete solver;

  return error;
  
} // test_isentropic_vortex


// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char *argv[])
{

  using namespace ppkMHD;
  
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
    
  }

  // default order to test
  int order = 2;

  // check command line for another order to test
  if (argc > 1) {
    int tmp = std::atoi(argv[1]);
    if (tmp >= 1 and tmp < 7)
      order = tmp;
  }

  std::array<real_t, 4> results;
  
  if (order==2) {

    std::array<int, 4> sizes={40, 80, 160, 320};
    
    // testing convergence for second order N=2
    for (std::size_t i = 0; i<sizes.size(); ++i)
      results[i] = test_isentropic_vortex<2,sdm::NORM_L1>(sizes[i],SSP_RK2);

    for (std::size_t i = 0; i<sizes.size(); ++i)
      printf("order %d, size=%4d, error=%6.4e\n",order,sizes[i],results[i]);
    
  } else if (order==3) {

    std::array<int, 4> sizes={20, 40, 80, 160};

    // testing convergence for order N=3
    for (std::size_t i = 0; i<sizes.size(); ++i)
      results[i] = test_isentropic_vortex<3,sdm::NORM_L1>(sizes[i],SSP_RK2);

    for (std::size_t i = 0; i<sizes.size(); ++i)
      printf("order %d, size=%4d, error=%6.4e\n",order,sizes[i],results[i]);
    
  } else if (order==4) {

    std::array<int, 4> sizes={10, 20, 40, 80};

    // testing convergence for order N=4
    for (std::size_t i = 0; i<sizes.size(); ++i)
      results[i] = test_isentropic_vortex<4,sdm::NORM_L1>(sizes[i],SSP_RK3);

    for (std::size_t i = 0; i<sizes.size(); ++i)
      printf("order %d, size=%4d, error=%6.4e\n",order,sizes[i],results[i]);
    
  }

  // save result in a numpy compatible file (for plotting with python / matplotlib)
  
  
#ifdef CUDA
  Kokkos::Cuda::finalize();
  Kokkos::HostSpace::execution_space::finalize();
#else
  Kokkos::finalize();
#endif
  
  return EXIT_SUCCESS;

} // end main

