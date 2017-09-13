/**
 * This executable is used to test velocity gradients functors.
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
#include "sdm/SDM_Interpolate_Functors.h"
#include "sdm/SDM_Flux_Functors.h"

// for IO
#include "utils/io/IO_Writer_SDM.h"

#include "test_sdm_gradient_velocity_init.h"

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
void test_gradient_velocity_functors()
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
  std::string input_file = dim == 2 ? "test_sdm_gradient_velocity_2D.ini" : "test_sdm_gradient_velocity_3D.ini";
  ConfigMap configMap(input_file);

  // create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // create solver
  sdm::SolverHydroSDM<dim,N> solver(params, configMap);

  // initialize the IO_Writer object (normally done in
  // SolverFactory's create method)
  solver.init_io_writer();

  int nbCells = dim==2 ?
    params.isize*params.jsize :
    params.isize*params.jsize*params.ksize;
  
  // init data
  {

    sdm::InitTestGradientVelocityFunctor<dim,N,TEST_DATA_VALUE, 0>
      functor(solver.params,
	      solver.sdm_geom,
	      solver.U);
    Kokkos::parallel_for(nbCells, functor);

      
    solver.save_solution();

  }


  //
  // velocity gradient X
  //
  {

    // create variables names for velocity gradients
    std::map<int, std::string> var_names_gradx;
    if (dim==2) {
      var_names_gradx[(int)VarIndexGrad2d::IGU] = "gradx_u";
      var_names_gradx[(int)VarIndexGrad2d::IGV] = "gradx_v";
    } else {
      var_names_gradx[(int)VarIndexGrad3d::IGU] = "gradx_u";
      var_names_gradx[(int)VarIndexGrad3d::IGV] = "gradx_v";
      var_names_gradx[(int)VarIndexGrad3d::IGW] = "gradx_w";
    }
    
    // create an io_writer
    auto io_writer =
      std::make_shared<ppkMHD::io::IO_Writer_SDM<dim,N>>(solver.params,
							 solver.configMap,
							 var_names_gradx,
							 solver.sdm_geom);

    // actual computation
    solver.template compute_velocity_gradients<IX>(solver.U, solver.Ugradx_v);
    
    DataArrayHost Ugradx_Host = Kokkos::create_mirror(solver.Ugradx_v);
    io_writer->save_data_impl(solver.Ugradx_v,
			      Ugradx_Host,
			      0,
			      0.0,
			      "Ugradx_v");
  }
  
  //
  // velocity gradient Y
  //
  {
    // create variables names for velocity gradients
    std::map<int, std::string> var_names_grady;
    if (dim==2) {
      var_names_grady[(int)VarIndexGrad2d::IGU] = "grady_u";
      var_names_grady[(int)VarIndexGrad2d::IGV] = "grady_v";
    } else {
      var_names_grady[(int)VarIndexGrad3d::IGU] = "grady_u";
      var_names_grady[(int)VarIndexGrad3d::IGV] = "grady_v";
      var_names_grady[(int)VarIndexGrad3d::IGW] = "grady_w";
    }
    
    // create an io_writer
    auto io_writer =
      std::make_shared<ppkMHD::io::IO_Writer_SDM<dim,N>>(solver.params,
							 solver.configMap,
							 var_names_grady,
							 solver.sdm_geom);

    // actual computation
    solver.template compute_velocity_gradients<IY>(solver.U, solver.Ugrady_v);
    
    DataArrayHost Ugrady_Host = Kokkos::create_mirror(solver.Ugrady_v);
    io_writer->save_data_impl(solver.Ugrady_v,
			      Ugrady_Host,
			      0,
			      0.0,
			      "Ugrady_v");
  }
  
  //
  // velocity gradient Z
  //
  if (dim==3) {
    // create variables names for velocity gradients
    std::map<int, std::string> var_names_gradz;
    if (dim==2) {
      var_names_gradz[(int)VarIndexGrad2d::IGU] = "gradz_u";
      var_names_gradz[(int)VarIndexGrad2d::IGV] = "gradz_v";
    } else {
      var_names_gradz[(int)VarIndexGrad3d::IGU] = "gradz_u";
      var_names_gradz[(int)VarIndexGrad3d::IGV] = "gradz_v";
      var_names_gradz[(int)VarIndexGrad3d::IGW] = "gradz_w";
    }
    
    // create an io_writer
    auto io_writer =
      std::make_shared<ppkMHD::io::IO_Writer_SDM<dim,N>>(solver.params,
							 solver.configMap,
							 var_names_gradz,
							 solver.sdm_geom);

    // actual computation
    solver.template compute_velocity_gradients<IZ>(solver.U, solver.Ugradz_v);
    
    DataArrayHost Ugradz_Host = Kokkos::create_mirror(solver.Ugradz_v);
    io_writer->save_data_impl(solver.Ugradz_v,
			      Ugradz_Host,
			      0,
			      0.0,
			      "Ugradz_v"); 
  }
  

} // test_gradient_velocity_functors

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
  std::cout << "==== Spectral Difference Method : =======================\n";
  std::cout << "==== velocity gradient test       =======================\n";
  std::cout << "=========================================================\n";

  // testing for multiple value of N in 2 to 6
  {
    // 2d
    test_gradient_velocity_functors<2,4>();

    // 3d
    test_gradient_velocity_functors<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
