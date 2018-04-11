/**
 * This executable is used to test sdm::SolverHydroSDM class, 
 * more specificly object EulerEquations.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/SolverHydroSDM.h"

#include "shared/EulerEquations.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
template<int dim, int N>
class TestFluxFunctor : public SDMBaseFunctor<dim,N> {
  
public:
  using typename SDMBaseFunctor<dim,N>::DataArray;

  using HydroState = typename ppkMHD::EulerEquations<dim>::HydroState;

  using eq = typename ppkMHD::EulerEquations<dim>;
  
  static constexpr auto dofMap = DofMap<dim,N>;
  
  TestFluxFunctor(HydroParams         params,
		  SDM_Geometry<dim,N> sdm_geom,
		  ppkMHD::EulerEquations<dim> euler,
		  DataArray           Udata,
		  DataArray           FluxData) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    euler(euler),
    Udata(Udata),
    FluxData(FluxData) {};

  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
     
    const real_t gamma0 = this->params.settings.gamma0;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // loop over cell DoF's
    for (int idy=0; idy<N; ++idy) {
      for (int idx=0; idx<N; ++idx) {
	
	HydroState q, flux;
	q[ID] = Udata(i  ,j  , dofMap(idx,idy,0,ID));
	q[IE] = Udata(i  ,j  , dofMap(idx,idy,0,IE));
	q[IU] = Udata(i  ,j  , dofMap(idx,idy,0,IU));
	q[IV] = Udata(i  ,j  , dofMap(idx,idy,0,IV));

	// compute kineti energy, internal energy and then pressure
	real_t eken = HALF_F * (q[IU]*q[IU] + q[IV]*q[IV]) / q[ID];
	real_t e_internal = q[IE] - eken;
	real_t pressure = (gamma0 - 1.0) * e_internal;
	
	eq::flux_x(q,pressure,flux);

	Udata(i  ,j  , dofMap(idx,idy,0,ID)) = flux[ID];
	Udata(i  ,j  , dofMap(idx,idy,0,IE)) = flux[IE];
	Udata(i  ,j  , dofMap(idx,idy,0,IU)) = flux[IU];
	Udata(i  ,j  , dofMap(idx,idy,0,IV)) = flux[IV];
		
      } // end for idx
    } // end for idy
    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
  } // end operator () - 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray Udata;
  DataArray FluxData;
  
}; // TestFluxFunctor

} // namespace sdm

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int dim,
	 int N>
void test_sdm_flux(int argc, char* argv[])
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
  solver.init_io();
  
  // init SDM geometry
  sdm::SDM_Geometry<dim,N> sdm_geom;
  sdm_geom.init(0);
  sdm_geom.init_lagrange_1d();

  // Euler equations
  ppkMHD::EulerEquations<dim> euler;
  
  // create test functor
  sdm::TestFluxFunctor<dim,N> functor(params, sdm_geom, euler, solver.U, solver.Uaux);
  Kokkos::parallel_for(solver.nbCells, functor);
  
  solver.save_solution();
  
} // test_sdm_io

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
    std::cout << "==== Spectral Difference Lagrange Flux test ====\n";
    std::cout << "================================================\n";
  }
  
  
  // testing for multiple values of N in 2 to 6
  {

    // 2d
    test_sdm_flux<2,4>(argc,argv);

    // 3d
    test_sdm_flux<3,4>(argc,argv);

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
