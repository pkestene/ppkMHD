/**
 * This executable is used to test sdm::SDM_Geometry class, 
 * more specific Lagrange interpolation.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <memory> // for std::unique_ptr / std::shared_ptr

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"
//#include "sdm/SolverHydroSDM.h"
#include "sdm/HydroInitFunctors.h"
#include "sdm/SDM_Interpolate_Functors.h"
#include "sdm/SDM_Compute_error.h"

#include "SDMTestFunctors.h"

// for IO
#include "utils/io/IO_ReadWrite_SDM.h"

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
bool test_lagrange_functor()
{

  using DataArray = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
  using DataArrayHost = typename DataArray::HostMirror;

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

  int isize = params.isize;
  int jsize = params.jsize;
  int ksize = params.ksize;

  // SDM config
  sdm::SDM_Geometry<dim,N> sdm_geom;
  sdm_geom.init(0);
  sdm_geom.init_lagrange_1d();

  // create solver
  //sdm::SolverHydroSDM<dim,N> solver(params, configMap);
  DataArray U = dim==2 ? 
    DataArray("U", isize*N, jsize*N, params.nbvar) :
    DataArray("U", isize*N, jsize*N, ksize*N, params.nbvar);
    
  DataArrayHost UHost = Kokkos::create_mirror(U);

  DataArray U2 = dim==2 ? 
    DataArray("U2", isize*N, jsize*N, params.nbvar) :
    DataArray("U2", isize*N, jsize*N, ksize*N, params.nbvar);
  
  DataArray Fluxes = dim==2 ?
    DataArray("Fluxes", isize*N, jsize*(N+1), params.nbvar) :
    DataArray("Fluxes", isize*N, jsize*(N+1), ksize*N, params.nbvar);
  
  DataArrayHost FluxHost = Kokkos::create_mirror(Fluxes);
  
  int nbDofsPerCell = (dim==2) ? N*N : N*N*N;

  int nbDofsPerCellFlux = (dim==2) ? N*(N+1) : N*N*(N+1);
  

  int nbDofs = dim==2 ? 
    nbDofsPerCell*params.isize*params.jsize : 
    nbDofsPerCell*params.isize*params.jsize*params.ksize;
  
  int nbDofsFlux = dim==2 ? 
    nbDofsPerCellFlux*params.isize*params.jsize : 
    nbDofsPerCellFlux*params.isize*params.jsize*params.ksize;
   
  std::map<int, std::string> m_variables_names;
  m_variables_names.clear();
  m_variables_names[ID] = "rho";
  m_variables_names[IP] = "energy";
  m_variables_names[IU] = "rho_vx"; // momentum component X
  m_variables_names[IV] = "rho_vy"; // momentum component Y
  if (dim==3)
    m_variables_names[IW] = "rho_vz"; // momentum component Z

  // create an io_writer
  auto io_writer =
    std::make_shared<ppkMHD::io::IO_ReadWrite_SDM<dim,N>>(params,
                                                          configMap,
                                                          m_variables_names,
                                                          sdm_geom);
  

  // init data
  {

    sdm::InitTestFunctor<dim,N,TEST_DATA_VALUE,0> functor(params,
							  sdm_geom,
							  U);
    Kokkos::parallel_for(nbDofs, functor);

    Kokkos::deep_copy(U2,U);

    // save initial condition
    io_writer->save_data_impl(U,
                              UHost,
                              0,
                              0.0,
                              "");
    
  }
  
  // call the interpolation functors
  {

    sdm::Interpolate_At_FluxPoints_Functor<dim,N,IY>::apply(params,
                                                            sdm_geom,
                                                            U,
                                                            Fluxes);

    io_writer-> template save_flux<IY>(Fluxes,
				       FluxHost,
				       0,
				       0.0);

  }

  
  {
    
    constexpr sdm::Interpolation_type_t interp = sdm::INTERPOLATE_SOLUTION_REGULAR;
    sdm::Interpolate_At_SolutionPoints_Functor<dim,
                                               N,
                                               IY,
                                               interp>::apply(params,
                                                              sdm_geom,
                                                              Fluxes,
                                                              U);
  }

  // compute L1 error between U and U2
  double error_accum = 0.0;
  if (dim==2) {

    for (int ivar = 0; ivar < params.nbvar; ++ivar) {
      real_t error_L1 = 
        sdm::Compute_Error_Functor_2d<N, sdm::NORM_L1>::apply(params,
                                                              sdm_geom,
                                                              U,
                                                              U2,
                                                              ivar,
                                                              nbDofs);
      printf("L1 error for variable %d : %e\n",ivar,error_L1);
      error_accum += error_L1;
    }

  } else if (dim==3) {
  
    for (int ivar = 0; ivar < params.nbvar; ++ivar) {
      real_t error_L1 = 
        sdm::Compute_Error_Functor_3d<N,sdm::NORM_L1>::apply(params,
                                                             sdm_geom,
                                                             U,
                                                             U2,
                                                             ivar,
                                                             nbDofs);
      printf("L1 error for variable %d : %e\n",ivar,error_L1);

    }

  }

  // compute difference between original data and after sol2flux / flux2sol
  // difference should be zero if original data is polynomial of
  // degree less than N, or just small
  {

    sdm::InitTestFunctor<dim,N,TEST_DATA_VALUE,1> functor(params,
							  sdm_geom,
							  U);
  
    Kokkos::parallel_for(nbDofs, functor);
    
    io_writer->save_data_impl(U,
                              UHost,
                              1,
                              1.0,
                              "");
    
  }

  return (error_accum < 1e-10);

} // test_lagrange_functor

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

  std::cout << "=========================================================\n";
  std::cout << "==== Spectral Difference Lagrange Interpolation test ====\n";
  std::cout << "=========================================================\n";

  bool passed = true;

  // testing for multiple value of N in 2 to 6
  {
    // 2d
    passed *= test_lagrange_functor<2,4>();

    // 3d
    passed *= test_lagrange_functor<3,4>();

  }

  Kokkos::finalize();

  // return 0 if all tests passed
  return (int) !passed;
  
}
