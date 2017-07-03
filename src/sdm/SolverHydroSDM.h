/**
 * Main class to drive a Spectral Difference Method numerical scheme.
 */
#ifndef SOLVER_HYDRO_SDM_H_
#define SOLVER_HYDRO_SDM_H_

#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"
#include "shared/BoundariesFunctors.h"
#include "shared/BoundariesFunctorsWedge.h"
#include "shared/initRiemannConfig2d.h"

// sdm
#include "sdm/SDM_Geometry.h"

// sdm functors (where the action takes place)
#include "sdm/HydroInitFunctors.h"
//#include "sdm/SDM_DtFunctor.h"
//#include "sdm/SDM_UpdateFunctors.h"

// for IO
#include "utils/io/IO_Writer_SDM.h"

// for specific init / border conditions
#include "shared/BlastParams.h"
#include "shared/KHParams.h"
#include "shared/WedgeParams.h"

namespace sdm {

/**
 * Main hydrodynamics data structure driving SDM numerical solver.
 *
 * \tparam dim dimension of the domain (2 or 3)
 * \tparam N is the number of solution points per direction
 *
 * The total number of solution points per cells is N^2 in 2D, and N^3 in 3D.
 * We use row-major format to enumerate DoF (degrees of freedom) inside cell,
 * i.e. we will define a mapping  
 * for example in 2D:
 *    index = DofMap(i,j,iv) computed as i+N*j+N*N*iv
 * DofMap returns index where one can find the variable iv (e.g. for density
 * just use ID) of the (i,j) Dof inside a given cell. the pair i,j identifies
 * a unique Dof among the N^2 Dof in 2D.
 * 
 */
template<int dim, int N>
class SolverHydroSDM : public ppkMHD::SolverBase
{

public:

  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim==2,DataArray2dHost,DataArray3dHost>::type;

  SolverHydroSDM(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroSDM();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroSDM<dim,N>* solver = new SolverHydroSDM<dim,N>(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  
  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray     U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes
  DataArray Fluxes_x, Fluxes_y, Fluxes_z;
  
  /*
   * Override base class method to initialize IO writer object
   */
  void init_io_writer();

  /*
   * SDM config
   */
  SDM_Geometry<dim,N> sdm_geom;
    
  /*
   * methods
   */

  //! initialize sdm (geometric terms matrix)
  void init_sdm_geometry();
    
  //! compute time step inside an MPI process, at shared memory level.
  double compute_dt_local();

  //! perform 1 time step (time integration).
  void next_iteration_impl();

  //! numerical scheme
  void time_integration(real_t dt);

  //! wrapper to tha actual time integation scheme
  void time_integration_impl(DataArray data_in, 
			     DataArray data_out, 
			     real_t dt);

  //! time integration using forward Euler method
  void time_int_forward_euler(DataArray data_in, 
			      DataArray data_out, 
			      real_t dt);

  //! time integration using SSP RK2
  void time_int_ssprk2(DataArray data_in, 
		       DataArray data_out, 
		       real_t dt);
  
  //! time integration using SSP RK3
  void time_int_ssprk3(DataArray data_in, 
		       DataArray data_out, 
		       real_t dt);

  //! time integration using SSP RK4
  void time_int_ssprk54(DataArray data_in, 
			DataArray data_out, 
			real_t dt);
  

  template<int dim_=dim>
  void make_boundaries(typename std::enable_if<dim_==2,DataArray2d>::type Udata);

  template<int dim_=dim>
  void make_boundaries(typename std::enable_if<dim_==3,DataArray3d>::type Udata);

  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_four_quadrant(DataArray Udata);
  void init_kelvin_helmholtz(DataArray Udata);
  void init_wedge(DataArray Udata);
  void init_isentropic_vortex(DataArray Udata);
  
  void save_solution_impl();

  // time integration
  bool forward_euler_enabled;
  bool ssprk2_enabled;
  bool ssprk3_enabled;
  bool ssprk54_enabled;
  
  int isize, jsize, ksize, nbCells;

}; // class SolverHydroSDM


// =======================================================
// ==== CLASS SolverHydroSDM IMPL =======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int N>
SolverHydroSDM<dim,N>::SolverHydroSDM(HydroParams& params,
				      ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), U2(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCells(params.isize*params.jsize),
  sdm_geom(),
  forward_euler_enabled(true),
  ssprk2_enabled(false),
  ssprk3_enabled(false),
  ssprk54_enabled(false)
{

  if (dim==3)
    nbCells = params.isize*params.jsize*params.ksize;
  
  m_nCells = nbCells;

  int nb_dof_per_cell = dim==2 ? N*N : N*N*N;
  int nb_dof = params.nbvar * nb_dof_per_cell;

  long long int total_mem_size = 0;
  
  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (dim==2) {

    U     = DataArray("U", isize, jsize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    U2    = DataArray("U2",isize, jsize, nb_dof);
    
    //Fluxes_x = DataArray("Fluxes_x", isize, jsize, nb_dof);
    //Fluxes_y = DataArray("Fluxes_y", isize, jsize, nb_dof);

    //total_mem_size += isize*jsize*nb_dof*4 * sizeof(real_t);
    //total_mem_size += isize*jsize * sizeof(real_t);
    //total_mem_size += isize*jsize*nb_dof * ncoefs * sizeof(real_t);
      
  } else if (dim==3) {

    U     = DataArray("U", isize, jsize, ksize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    U2    = DataArray("U2",isize, jsize, ksize, nb_dof);
    
    //Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nb_dof);
    //Fluxes_y = DataArray("Fluxes_y", isize, jsize, ksize, nb_dof);
    //Fluxes_z = DataArray("Fluxes_z", isize, jsize, ksize, nb_dof);

    //total_mem_size += isize*jsize*ksize*nb_dof*5 * sizeof(real_t);
    //total_mem_size += isize*jsize*ksize * sizeof(real_t);
    //total_mem_size += isize*jsize*ksize*nb_dof * ncoefs * sizeof(real_t);

  }

  /*
   * Init Spectral Difference Mmethod geometry
   * (solution + flux points locations)
   */
  init_sdm_geometry();
    
  /*
   * Time integration
   */
  forward_euler_enabled = configMap.getBool("sdm", "forward_euler", true);
  ssprk2_enabled        = configMap.getBool("sdm", "ssprk2", false);
  ssprk3_enabled        = configMap.getBool("sdm", "ssprk3", false);
  ssprk54_enabled       = configMap.getBool("sdm", "ssprk54", false);

  if (ssprk2_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * sizeof(real_t);
    }
    
  } else if (ssprk3_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * 2 * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * 2 * sizeof(real_t);
    }
    
  } else if (ssprk54_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, nb_dof);
      U_RK3 = DataArray("U_RK3",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * 3 * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, ksize, nb_dof);
      U_RK3 = DataArray("U_RK3",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * 3 * sizeof(real_t);
    }
    
  }

  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  // } else if ( !m_problem_name.compare("blast") ) {

  //   init_blast(U);

  // } else if ( !m_problem_name.compare("four_quadrant") ) {

  //   init_four_quadrant(U);

  // } else if ( !m_problem_name.compare("kelvin-helmholtz") or
  // 	      !m_problem_name.compare("kelvin_helmholtz")) {

  //   init_kelvin_helmholtz(U);

  // } else if ( !m_problem_name.compare("wedge") ) {

  //   init_wedge(U);
    
  // } else if ( !m_problem_name.compare("isentropic_vortex") ) {

  //   init_isentropic_vortex(U);
    
  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - Four Quadrant" << std::endl;
    init_implode(U);
    //init_four_quadrant(U);

  }

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (myRank==0) {
    std::cout << "##########################" << "\n";
    std::cout << "Solver is " << m_solver_name << "\n";
    std::cout << "Problem (init condition) is " << m_problem_name << "\n";
    std::cout << "Spectral Difference order (N) : " << N << "\n";
    std::cout << "Time integration is :\n";
    std::cout << "Forward Euler : " << forward_euler_enabled << "\n";
    std::cout << "SSPRK2        : " << ssprk2_enabled << "\n";
    std::cout << "SSPRK3        : " << ssprk3_enabled << "\n";
    std::cout << "SSPRK54       : " << ssprk54_enabled << "\n";
    std::cout << "##########################" << "\n";
    
    // print parameters on screen
    params.print();
    std::cout << "##########################" << "\n";
    std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
    std::cout << "##########################" << "\n";
  }
  
  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);
  
} // SolverHydroSDM::SolverHydroSDM

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int N>
SolverHydroSDM<dim,N>::~SolverHydroSDM()
{

} // SolverHydroSDM::~SolverHydroSDM

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_io_writer()
{
  
  // install a new IO_Writer sdm-specific
  m_io_writer = std::make_shared<ppkMHD::io::IO_Writer_SDM<dim,N>>(params,
								   configMap,
								   m_variables_names,
								   sdm_geom);

} // SolverHydroSDM<dim,N>::init_io_writer

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_sdm_geometry()
{

  // TODO
  sdm_geom.init(0);
  sdm_geom.init_lagrange_1d();
  
} // SolverHydroSDM<dim,N>::init_sdm_geometry

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template<int dim, int N>
double SolverHydroSDM<dim,N>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  // typedef computeDtFunctor
  // using ComputeDtFunctor =
  //   typename std::conditional<dim==2,
  // 			      ComputeDtFunctor2d<N>,
  // 			      ComputeDtFunctor3d<N>>::type;

  // // call device functor
  // ComputeDtFunctor computeDtFunctor(params, monomialMap.data, Udata);
  // Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  // rescale dt to match the space order N+1
  if (N >= 2 and ssprk3_enabled)
    dt = pow(dt, (N+1.0)/3.0);
  
  return dt;

} // SolverHydroSDM::compute_dt_local

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::next_iteration_impl()
{

  if (m_iteration % 10 == 0) {
    //std::cout << "time step=" << m_iteration << " (dt=" << m_dt << ")" << std::endl;
    printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iteration,m_dt, m_t);
  }
  
  // output
  if (params.enableOutput) {
    if ( should_save_solution() ) {
      
      std::cout << "Output results at time t=" << m_t
		<< " step " << m_iteration
		<< " dt=" << m_dt << std::endl;
      
      save_solution();
      
    } // end output
  } // end enable output
  
  // compute new dt
  timers[TIMER_DT]->start();
  compute_dt();
  timers[TIMER_DT]->stop();
  
  // perform one step integration
  time_integration(m_dt);
  
} // SolverHydroSDM::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_integration(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    time_integration_impl(U , U2, dt);
  } else {
    time_integration_impl(U2, U , dt);
  }
  
} // SolverHydroSDM::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of SDM scheme
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_integration_impl(DataArray data_in, 
						  DataArray data_out, 
						  real_t dt)
{
  
  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  if (ssprk2_enabled) {
    
    time_int_ssprk2(data_in, data_out, dt);
    
  } else if (ssprk3_enabled) {
    
    time_int_ssprk3(data_in, data_out, dt);
    
  } else if (ssprk54_enabled) {
    
    time_int_ssprk54(data_in, data_out, dt);
    
  } else {
    
    time_int_forward_euler(data_in, data_out, dt);
    
  }
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroSDM::time_integration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Forward Euler time integration
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_forward_euler(DataArray data_in, 
						   DataArray data_out, 
						   real_t dt)
{
  
  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;
    
} // SolverHydroSDM::time_int_forward_euler

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK2 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 2th order.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * SSP-RK22 (2 stages, 2nd order).
 *
 * The cfl coefficient is 1, i.e.
 *
 * Dt <= cfl Dt_FE
 * where Dt_FE is the forward Euler Dt
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk2(DataArray data_in, 
						  DataArray data_out, 
						  real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);
  
  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================

  // ==================================================================
  // second step : U_{n+1} = 0.5 * (U_n + U_RK1 + dt * fluxes(U_RK1) )
  // ==================================================================
  
} // SolverHydroSDM::time_int_ssprk2

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK3 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 3th order.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * SSP-RK33 (3 stages, 3nd order).
 *
 * Note: This scheme is also call TVD-RK3
 *
 * The cfl coefficient is 1, i.e.
 *
 * Dt <= cfl Dt_FE
 * where Dt_FE is the forward Euler Dt
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk3(DataArray data_in, 
						  DataArray data_out, 
						  real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);

  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================
  
  // ========================================================================
  // second step : U_RK2 = 3/4 * U_n + 1/4 * U_RK1 + 1/4 * dt * fluxes(U_RK1)
  // ========================================================================

  // ============================================================================
  // thrird step : U_{n+1} = 1/3 * U_n + 2/3 * U_RK2 + 2/3 * dt * fluxes(U_RK2)
  // ============================================================================

} // SolverHydroSDM::time_int_ssprk3

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK54 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 4th order, 5 stages.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * This scheme is call SSP-RK54
 * 
 * It has been proved that no 4th order RK, 4 stages SSP-RK scheme
 * exists with positive coefficients (Goettlib and Shu, Total variation
 * diminishing Runge-Kutta schemes, Math. Comp., 67 (1998), pp. 73–85.).
 * This means a SSP-RK44 scheme will have negative coefficients, and we need to
 * have a flux operator backward in time stable.
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk54(DataArray data_in, 
						   DataArray data_out, 
						   real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);

  std::cout << "SSP-RK54 is currently unimplemented\n";
  
} // SolverHydroSDM::time_int_ssprk54

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim, int N>
template<int dim_>
void SolverHydroSDM<dim,N>::make_boundaries(typename std::enable_if<dim_==2,DataArray2d>::type Udata)
{
  
  bool mhd_enabled = false;

  make_boundaries_serial(Udata, mhd_enabled);
  
} // SolverHydroSDM::make_boundaries

template<int dim, int N>
template<int dim_>
void SolverHydroSDM<dim,N>::make_boundaries(typename std::enable_if<dim_==3,DataArray3d>::type Udata)
{

  bool mhd_enabled = false;

  make_boundaries_serial(Udata, mhd_enabled);

} // SolverHydroSDM::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_implode(DataArray Udata)
{

  InitImplodeFunctor<dim,N> functor(params, sdm_geom, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_blast(DataArray Udata)
{

  // BlastParams blastParams = BlastParams(configMap);
  
  // InitBlastFunctor<dim,N> functor(params, monomialMap.data, blastParams, Udata);
  // Kokkos::parallel_for(nbCells, functor);

} // SolverHydroSDM::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant  riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_four_quadrant(DataArray Udata)
{

  // int configNumber = configMap.getInteger("riemann2d","config_number",0);
  // real_t xt = configMap.getFloat("riemann2d","x",0.8);
  // real_t yt = configMap.getFloat("riemann2d","y",0.8);
    
  // HydroState2d U0, U1, U2, U3;
  // ppkMHD::getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  // ppkMHD::primToCons_2D(U0, params.settings.gamma0);
  // ppkMHD::primToCons_2D(U1, params.settings.gamma0);
  // ppkMHD::primToCons_2D(U2, params.settings.gamma0);
  // ppkMHD::primToCons_2D(U3, params.settings.gamma0);
  
  // InitFourQuadrantFunctor<dim,N> functor(params, monomialMap.data,
  // 					      Udata,
  // 					      U0, U1, U2, U3,
  // 					      xt, yt);
  // Kokkos::parallel_for(nbCells, functor);
    
} // init_four_quadrant

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability test.
 * 
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_kelvin_helmholtz(DataArray Udata)
{

  // KHParams khParams = KHParams(configMap);

  // InitKelvinHelmholtzFunctor<dim,N> functor(params,
  // 						 monomialMap.data,
  // 						 khParams,
  // 						 Udata);
  // Kokkos::parallel_for(nbCells, functor);

} // SolverHydroSDM::init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 * 
 * 
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_wedge(DataArray Udata)
{

  // WedgeParams wparams(configMap, 0.0);
  
  // InitWedgeFunctor<dim,N> functor(params, monomialMap.data, wparams, Udata);
  // Kokkos::parallel_for(nbCells, functor);
  
} // init_wedge

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_isentropic_vortex(DataArray Udata)
{

  // IsentropicVortexParams iparams(configMap);

  // InitIsentropicVortexFunctor<dim,N> functor(params, monomialMap.data, iparams, Udata);
  // Kokkos::parallel_for(nbCells, functor);
  
} // init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved, m_t);
  else
    save_data(U2, Uhost, m_times_saved, m_t);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroSDM::save_solution_impl()

} // namespace sdm

#endif // SOLVER_HYDRO_SDM_H_
