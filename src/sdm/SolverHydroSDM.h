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
#include "shared/EulerEquations.h"

// sdm
#include "sdm/SDM_Geometry.h"

// sdm functors (where the action takes place)
#include "sdm/HydroInitFunctors.h"
#include "sdm/SDM_Dt_Functor.h"
#include "sdm/SDM_Interpolate_Functors.h"
#include "sdm/SDM_Flux_Functors.h"
#include "sdm/SDM_Flux_with_Limiter_Functors.h"
#include "sdm/SDM_Run_Functors.h"
#include "sdm/SDM_Boundaries_Functors.h"
#include "sdm/SDM_Limiter_Functors.h"
#include "sdm/SDM_Positivity_preserving.h"

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
 * Time integration is configurable through parameter file. Allowed
 * possiblities are foward_euler, ssprk2 or ssprk3.
 * 
 * Shock capturing with limiters is a delicate subject.
 * It is disabled by default, but can be enable through parameter
 * flux_limiting_enabled.
 *
 * Our first idea was to implement adapt the work by
 * May, Jameson, "A Spectral Difference Method for the Euler
 * and Navier-Stokes Equations on Unstructured Meshes", AIAA 2006-304
 * http://aero-comlab.stanford.edu/Papers/may.aiaa.06-0304.pdf
 * May/Jameson limiting procedure consists in:
 * 1. for each cell, compute reference state as average HydroState 
 *    (stored in Uaverage)
 * 2. for each cell, compute Umin, Umax as min max HydroState in a neighborhood
 * 3. for each cell, evaluate is the high-order cell-border reconstruction 
 *    must be demoted to linear reconstruction. If any of the 
 *    K=number_of_faces x N reconstructed state violates the TVD criterium,
 *    all the faces reconstruction will be demoted to linear reconstruction.
 *
 * We finaly implement the original idea published in Cockburn and Shu,
 * "The Runge-Kutta Discontinuous Galerkin Method for Conservation Laws V: 
 * MultiDimensinal systems", Journal of Computational Physics, 141, 199-224 (1998).
 *
 * 1. compute cell-average of conservative variables, as well as cell-aveage gradient
 * of the conservative variables.
 * 2. for each space direction, for each cell, compute 3 state-vector:
 *    a. current cell average gradient times dx
 *    b. backward difference of cell-averaged value (current and neighbor cell)
 *    c. forward  difference of cell-averaged value (current and neighbor cell)
 * 3. project these 3 vectors in the local characteristics space (eigenspace of the
 *    local flux Jacobian matrix); 
 * 4. for each component, perform a TVB-modified minmod limiting of the 3 values 
 *    to detect if dofs must be modified to a 1st order polynomial approximation
 * 5. if limiting detection, was positive actually perform the 1st order modification
 *    in current cell.
 */
template<int dim, int N>
class SolverHydroSDM : public ppkMHD::SolverBase
{

public:

  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim==2,DataArray2dHost,DataArray3dHost>::type;

  //! a type to store some coefficients needed to perform Runge-Kutta integration
  using coefs_t = Kokkos::Array<real_t,3>;

  static constexpr int get_dim() {return dim;};
  static constexpr int get_N()   {return N;};
  
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

  /**
   * Return a string that uniquely identifies a SDM Solver. This class
   * returns a string <tt>SolverHydroSDM<dim,N></tt>, with @p dim and @p N
   * replaced by appropriate template values.
   */
  virtual std::string get_name () const;

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     Uaux;  /*!< auxiliary hydrodynamics conservative variables arrays (used in computing fluxes divergence */

  DataArray     Uaverage; /*! used if limiting is enabled */
  DataArray     Umin;     /*! used if limiting is enabled */
  DataArray     Umax;     /*! used if limiting is enabled */
  
  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray     U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes
  DataArray Fluxes;
  
  /*
   * Override base class method to initialize IO writer object
   */
  void init_io_writer();

  //! SDM config
  SDM_Geometry<dim,N> sdm_geom;
    
  //! system of equations
  ppkMHD::EulerEquations<dim> euler;

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
  void time_integration_impl(DataArray Udata, 
			     DataArray Udata_fdiv, 
			     real_t dt);

  //! compute flux divergence, the main term to perform the actual update
  //! in one of the Runge-Kutta methods.
  //! Udata is used in input (one of Runge-Kutta sub-step data)
  //! Udata_fdiv is used in,out as an accumulator
  //! so that the actual update will be U_{n+1}=U_{n}-dt*Udata_fdiv
  //! this operator is used in every Runge Kutta time intergrator
  void compute_fluxes_divergence(DataArray Udata, 
				 DataArray Udata_fdiv, 
				 real_t    dt);

  //! time integration using forward Euler method
  void time_int_forward_euler(DataArray Udata, 
			      DataArray Udata_fdiv, 
			      real_t dt);

  //! time integration using SSP RK2
  void time_int_ssprk2(DataArray Udata, 
		       DataArray Udata_fdiv, 
		       real_t dt);
  
  //! time integration using SSP RK3
  void time_int_ssprk3(DataArray Udata, 
		       DataArray Udata_fdiv, 
		       real_t dt);

  //! time integration using SSP RK4
  void time_int_ssprk54(DataArray Udata, 
			DataArray Udata_fdiv, 
			real_t dt);

  //! erase a solution data array
  void erase(DataArray data);

  //! avoid override the base class make_boundary method
  void make_boundary_sdm(DataArray  Udata,
			 FaceIdType faceId,
			 bool       mhd_enabled);

  void make_boundaries(DataArray Udata);

  void make_boundaries_sdm_serial(DataArray Udata, bool mhd_enabled);
  
  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_four_quadrant(DataArray Udata);
  void init_kelvin_helmholtz(DataArray Udata);
  void init_wedge(DataArray Udata);
  void init_isentropic_vortex(DataArray Udata);
  
  void save_solution_impl();

  //! debug routine that saves a flux data array (for a given direction)
  // template <int dir>
  // void save_flux();

  //! flux limiting procedure
  bool flux_limiting_enabled;
  
  //! time integration
  bool forward_euler_enabled;
  bool ssprk2_enabled;
  bool ssprk3_enabled;
  bool ssprk54_enabled;

  //! limiter (for shock capturing features)
  bool limiter_enabled;

  //! positivity preserving (density + pressure)
  bool positivity_enabled;

  
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
  U(), Uhost(), Uaux(),
  Fluxes(), 
  sdm_geom(),
  flux_limiting_enabled(false),
  forward_euler_enabled(true),
  ssprk2_enabled(false),
  ssprk3_enabled(false),
  ssprk54_enabled(false),
  limiter_enabled(false),
  positivity_enabled(false),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCells(params.isize*params.jsize)
{

  solver_type = SOLVER_SDM;

  if (dim==3)
    nbCells = params.isize*params.jsize*params.ksize;
  
  m_nCells = nbCells;

  int nb_dof_per_cell = dim==2 ? N*N : N*N*N;
  int nb_dof = params.nbvar * nb_dof_per_cell;

  m_nDofsPerCell = nb_dof_per_cell;
  
  // useful for allocating Fluxes, for conservative variables at flux points
  int nb_dof_flux = dim==2 ? (N+1)*N*params.nbvar : (N+1)*N*N*params.nbvar;
  
  long long int total_mem_size = 0;
  
  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (dim==2) {

    U     = DataArray("U", isize, jsize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    Uaux  = DataArray("Uaux",isize, jsize, nb_dof);
    
    Fluxes = DataArray("Fluxes", isize, jsize, nb_dof_flux);

    total_mem_size += isize*jsize*nb_dof      * sizeof(real_t); // U
    total_mem_size += isize*jsize*nb_dof      * sizeof(real_t); // Uaux
    total_mem_size += isize*jsize*nb_dof_flux * sizeof(real_t); // Fluxes
    
  } else if (dim==3) {

    U     = DataArray("U", isize, jsize, ksize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    Uaux  = DataArray("Uaux",isize, jsize, ksize, nb_dof);
    
    Fluxes = DataArray("Fluxes", isize, jsize, ksize, nb_dof_flux);

    total_mem_size += isize*jsize*ksize*nb_dof      * sizeof(real_t); // U
    total_mem_size += isize*jsize*ksize*nb_dof      * sizeof(real_t); // Uaux
    total_mem_size += isize*jsize*ksize*nb_dof_flux * sizeof(real_t); // Fluxes

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
   * limiter
   */
  limiter_enabled = configMap.getBool("sdm", "limiter_enabled", false);
  if (limiter_enabled) {
    
    if (dim==2) {
      Umin     = DataArray("Umin"    ,isize,jsize,params.nbvar);
      Umax     = DataArray("Umax"    ,isize,jsize,params.nbvar);
    } else if (dim==3) {
      Umin     = DataArray("Umin"    ,isize,jsize,ksize,params.nbvar);
      Umax     = DataArray("Umax"    ,isize,jsize,ksize,params.nbvar);
    }
  }

  /*
   * Data arrary Uaverage is used in both positivity preserving and
   * limiter
   */
  positivity_enabled = configMap.getBool("sdm", "positivity_enabled", false);
  if (positivity_enabled or limiter_enabled) {

    if (dim==2) {
      Uaverage = DataArray("Uaverage",isize,jsize,params.nbvar);
    } else if (dim==3) {
      Uaverage = DataArray("Uaverage",isize,jsize,ksize,params.nbvar);
    }
    
  }
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(U);

  } else if ( !m_problem_name.compare("four_quadrant") ) {

    init_four_quadrant(U);

  // } else if ( !m_problem_name.compare("kelvin-helmholtz") or
  // 	      !m_problem_name.compare("kelvin_helmholtz")) {

  //   init_kelvin_helmholtz(U);

  } else if ( !m_problem_name.compare("wedge") ) {
    
    init_wedge(U);
    
  } else if ( !m_problem_name.compare("isentropic_vortex") ) {

    init_isentropic_vortex(U);
    
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
std::string SolverHydroSDM<dim,N>::get_name() const
{

  std::ostringstream buf;
  buf << "SolverHydroSDM<"
      << dim << "," << N << ">";

  return buf.str();
  
} // SolverHydroSDM<dim,N>::get_name

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
 * \return dt time step (local to current MPI process)
 *
 * \note
 * The global time step is computed in compute_dt (from base class SolverBase)
 * which actually calls compute_dt_local.
 *
 */
template<int dim, int N>
double SolverHydroSDM<dim,N>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  Udata = U;
  
  using ComputeDtFunctor =
    typename std::conditional<dim==2,
  			      ComputeDt_Functor_2d<N>,
  			      ComputeDt_Functor_3d<N>>::type;
  
  // call device functor
  ComputeDtFunctor computeDtFunctor(params, sdm_geom, euler, Udata);
  Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
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

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI
  if (myRank==0) {
    if (m_iteration % 10 == 0) {
      printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iteration,m_dt, m_t);
    }
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
  
  time_integration_impl(U , Uaux, dt);
  
} // SolverHydroSDM::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of SDM scheme
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_integration_impl(DataArray Udata, 
						  DataArray Udata_fdiv,
						  real_t dt)
{
  
  // fill ghost cell in Udata
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(Udata);
  timers[TIMER_BOUNDARIES]->stop();
      
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  if (ssprk2_enabled) {
    
    time_int_ssprk2(Udata, Udata_fdiv, dt);

  } else if (ssprk3_enabled) {
    
    time_int_ssprk3(Udata, Udata_fdiv, dt);
    
  } else if (ssprk54_enabled) {
    
    time_int_ssprk54(Udata, Udata_fdiv, dt);
    
  } else {
    
    time_int_forward_euler(Udata, Udata_fdiv, dt);
    
  }
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroSDM::time_integration_impl

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////
// Compute fluxes divergence (Udata_fdiv)
// Taken the conservation law dU/dt + div F(U) = 0
// where div F(U) = dF/dx + dG/dy + dH/dz, and F(U) is a short
// notation for invicid fluxes + viscous terms + gravitational terms ...
//
// the actual update will be U_{n+1}=U_{n}-dt*Udata_fdiv
// //////////////////////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::compute_fluxes_divergence(DataArray Udata, 
						      DataArray Udata_fdiv,
						      real_t dt)
{

  // Here is the plan:
  // for each direction
  //  1. interpolate conservative variables from solution points to flux points
  //  2. compute inplace flux at flux points (interior + cell borders)
  //     (TODO: apply any flux limiter ?)
  //  3. compute viscous flux + source terms
  //  4. evaluate flux derivatives at solution points and accumulate in Udata_fdiv

  // erase Udata_fdiv
  erase(Udata_fdiv);
 
  // if limiter or positivity preserving are enabled,
  // we first compute Uaverage (cell volume averaged conservative variables) 
  if (limiter_enabled or positivity_enabled) {
    // compute Uaverage
    {
      Average_Conservative_Variables_Functor<dim,N> functor(params,
							    sdm_geom,
							    Udata,
							    Uaverage);
      Kokkos::parallel_for(nbCells, functor);
    }
  } // end limiter_enabled

  // if limiter is enabled we need to access cell neighborhood for min/max
  // cell average values
  if (limiter_enabled) {
    // compute Umin / Umax
    {
      MinMax_Conservative_Variables_Functor<dim,N> functor(params,
							   sdm_geom,
							   Uaverage,
							   Umin,
							   Umax);
      Kokkos::parallel_for(nbCells, functor);
    }
  }
  
  if (positivity_enabled) {
    Apply_positivity_Functor_v2<dim,N> functor(params,
					       sdm_geom,
					       Udata,
					       Uaverage);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  //
  // Dir X
  //
  {
    // 1. interpolate conservative variables from solution points to flux points
    {
      
      Interpolate_At_FluxPoints_Functor<dim,N,IX> functor(params,
							  sdm_geom,
							  Udata,
							  Fluxes);
      Kokkos::parallel_for(nbCells, functor);
      
    }

    // 1.1 ensure positivity (density and pressure),
    if (positivity_enabled) {
      if (0) {
	Apply_positivity_Functor<dim,N,IX> functor(params,
						   sdm_geom,
						   Udata,
						   Fluxes,
						   Uaverage);
	Kokkos::parallel_for(nbCells, functor);
      }      
    }
    
    // 2. inplace computation of fluxes along X direction at flux points
    if (limiter_enabled) {
      // compute flux at interior flux points +
      // reconstructed state at cell borders
      {
	Compute_Reconstructed_state_with_Limiter_Functor<dim,N,IX> functor(params,
									   sdm_geom,
									   euler,
									   Udata,
									   Uaverage,
									   Umin, Umax,
									   Fluxes);
	Kokkos::parallel_for(nbCells, functor);
      }
    } // end limiter enabled

    {
      ComputeFluxAtFluxPoints_Functor<dim,N,IX> functor(params,
							sdm_geom,
							euler,
							Fluxes);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 3. viscous terms + source terms (TODO)

    // 4. compute derivative and accumulate in Udata_fdiv
    {

      Interpolate_At_SolutionPoints_Functor<dim,N,IX> functor(params,
							      sdm_geom,
							      Fluxes,
							      Udata_fdiv);
      Kokkos::parallel_for(nbCells, functor);
      
    }
    
  } // end dir X

  //
  // Dir Y
  //
  {
    // 1. interpolate conservative variables from solution points to flux points
    {
      
      Interpolate_At_FluxPoints_Functor<dim,N,IY> functor(params,
							  sdm_geom,
							  Udata,
							  Fluxes);
      Kokkos::parallel_for(nbCells, functor);
      
    }
    
    // 1.1 ensure positivity (density and pressure)
    if (positivity_enabled) {
      if (0) {
	Apply_positivity_Functor<dim,N,IY> functor(params,
						   sdm_geom,
						   Udata,
						   Fluxes,
						   Uaverage);
	Kokkos::parallel_for(nbCells, functor);
      }
    }

    // 2. inplace computation of fluxes along Y direction at flux points
    if (limiter_enabled) {
      // compute flux at interior flux points +
      // reconstructed state at cell borders
      {
	Compute_Reconstructed_state_with_Limiter_Functor<dim,N,IY> functor(params,
									   sdm_geom,
									   euler,
									   Udata,
									   Uaverage,
									   Umin, Umax,
									   Fluxes);
	Kokkos::parallel_for(nbCells, functor);
      }
    } // end limiter enabled

    {
      ComputeFluxAtFluxPoints_Functor<dim,N,IY> functor(params,
							sdm_geom,
							euler,
							Fluxes);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 3. viscous terms + source terms (TODO)

    // 4. compute derivative and accumulate in Udata_fdiv
    {

      Interpolate_At_SolutionPoints_Functor<dim,N,IY> functor(params,
							      sdm_geom,
							      Fluxes,
							      Udata_fdiv);
      Kokkos::parallel_for(nbCells, functor);
      
    }
    
  } // end dir Y

  
  if (dim == 3) {
    //
    // Dir Z
    //
    {
      // 1. interpolate conservative variables from solution points to flux points
      {
	
	Interpolate_At_FluxPoints_Functor<dim,N,IZ> functor(params,
							    sdm_geom,
							    Udata,
							    Fluxes);
	Kokkos::parallel_for(nbCells, functor);
	
      }
      
      // 1.1 ensure positivity (density and pressure)
      if (positivity_enabled) {
	if (0) {
	  Apply_positivity_Functor<dim,N,IZ> functor(params,
						     sdm_geom,
						     Udata,
						     Fluxes,
						   Uaverage);
	  Kokkos::parallel_for(nbCells, functor);
	}
      }
      
      // 2. inplace computation of fluxes along Z direction at flux points
      if (limiter_enabled) {
	// compute flux at interior flux points +
	// reconstructed state at cell borders
	{
	  Compute_Reconstructed_state_with_Limiter_Functor<dim,N,IZ> functor(params,
									     sdm_geom,
									     euler,
									     Udata,
									     Uaverage,
									     Umin, Umax,
									     Fluxes);
	  Kokkos::parallel_for(nbCells, functor);
	}
      } // end limiter enabled
      
      {
	ComputeFluxAtFluxPoints_Functor<dim,N,IZ> functor(params,
							  sdm_geom,
							  euler,
							  Fluxes);
	Kokkos::parallel_for(nbCells, functor);
      }
      
      // 3. viscous terms + source terms (TODO)
      
      // 4. compute derivative and accumulate in Udata_fdiv
      {
	
	Interpolate_At_SolutionPoints_Functor<dim,N,IZ> functor(params,
								sdm_geom,
								Fluxes,
								Udata_fdiv);
	Kokkos::parallel_for(nbCells, functor);
	
      }
      
    } // end dir Z
    
  } // end dim == 3
  
  
} // SolverHydroSDM<dim,N>::compute_fluxes_divergence

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Forward Euler time integration
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_forward_euler(DataArray Udata, 
						   DataArray Udata_fdiv,
						   real_t dt)
{
    
  // evaluate flux divergence
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
  
  // perform actual time update in place in Udata: U_{n+1} = U_{n} - dt * Udata_fdiv
  // translated into Udata = 1.0*Udata + 0.0*Udata - dt * Udata_fdiv 
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
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
void SolverHydroSDM<dim,N>::time_int_ssprk2(DataArray Udata, 
					    DataArray Udata_fdiv, 
					    real_t dt)
{

  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
    
  // perform actual time update : U_RK1 = 1.0 * U_{n} + 0.0 * U_{n} - dt * Udata_fdiv
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK1, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ================================================================
  // second step :
  // U_{n+1} = 0.5 * U_n + 0.5 * U_RK1 - 0.5 * dt * div_fluxes(U_RK1)
  // ================================================================
  make_boundaries(U_RK1);
  compute_fluxes_divergence(U_RK1, Udata_fdiv, dt);

  {
    coefs_t coefs= {0.5, 0.5, -0.5};    
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, U_RK1, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
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
void SolverHydroSDM<dim,N>::time_int_ssprk3(DataArray Udata, 
					    DataArray Udata_fdiv, 
					    real_t dt)
{

  // ===============================================
  // first stage : U_RK1 = U_n - dt * div_fluxes(U_n)
  // ===============================================
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
  
  // perform : U_RK1 = 1.0 * U_{n} + 0.0 * U_{n} - dt * Udata_fdiv 
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK1, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ==============================================================
  // second stage :
  // U_RK2 = 3/4 * U_n + 1/4 * U_RK1 - 1/4 * dt * div_fluxes(U_RK1)
  // ==============================================================
  make_boundaries(U_RK1);
  compute_fluxes_divergence(U_RK1, Udata_fdiv, dt);
  {
    coefs_t coefs = {0.75, 0.25, -0.25};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK2, Udata, U_RK1, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // ================================================================
  // third stage :
  // U_{n+1} = 1/3 * U_n + 2/3 * U_RK2 - 2/3 * dt * div_fluxes(U_RK2)
  // ================================================================
  make_boundaries(U_RK2);
  compute_fluxes_divergence(U_RK2, Udata_fdiv, dt);
  {
    coefs_t coefs = {1.0/3, 2.0/3, -2.0/3};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, U_RK2, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

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
void SolverHydroSDM<dim,N>::time_int_ssprk54(DataArray Udata, 
					     DataArray Udata_fdiv, 
					     real_t dt)
{

  UNUSED(Udata);
  UNUSED(Udata_fdiv);
  UNUSED(dt);

  std::cout << "SSP-RK54 is currently unimplemented\n";
  
} // SolverHydroSDM::time_int_ssprk54

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::erase(DataArray data)
{

  SDM_Erase_Functor<dim,N> functor(params, sdm_geom, data);
  
  Kokkos::parallel_for(nbCells, functor);
  
} // SolverHydroSDM<dim,N>::erase

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundary_sdm(DataArray   Udata,
					      FaceIdType faceId,
					      bool mhd_enabled)
{

  UNUSED(mhd_enabled);
  
  const int ghostWidth=params.ghostWidth;
  int max_size = std::max(params.isize,params.jsize);
  int nbIter = ghostWidth * max_size;

  if (dim==3) {
    max_size = std::max(max_size,params.ksize);
    nbIter = ghostWidth * max_size * max_size;
  }

  if (faceId == FACE_XMIN) {

    MakeBoundariesFunctor_SDM<dim,N,FACE_XMIN> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }

  if (faceId == FACE_XMAX) {

    MakeBoundariesFunctor_SDM<dim,N,FACE_XMAX> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }
  
  if (faceId == FACE_YMIN) {

    MakeBoundariesFunctor_SDM<dim,N,FACE_YMIN> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }

  if (faceId == FACE_YMAX) {

    MakeBoundariesFunctor_SDM<dim,N,FACE_YMAX> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }

  if (dim == 3) {
    if (faceId == FACE_ZMIN) {
      
      MakeBoundariesFunctor_SDM<dim,N,FACE_ZMIN> functor(params, sdm_geom, Udata);
      Kokkos::parallel_for(nbIter, functor);
      
    }
    
    if (faceId == FACE_ZMAX) {
      
      MakeBoundariesFunctor_SDM<dim,N,FACE_ZMAX> functor(params, sdm_geom, Udata);
      Kokkos::parallel_for(nbIter, functor);
      
    }
  }
  
} // SolverHydroSDM<dim,N>::make_boundary

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundaries(DataArray Udata)
{
  
  bool mhd_enabled = false;

  make_boundaries_sdm_serial(Udata, mhd_enabled);
  
} // SolverHydroSDM::make_boundaries

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundaries_sdm_serial(DataArray Udata,
						       bool mhd_enabled)
{

  make_boundary_sdm(Udata, FACE_XMIN, mhd_enabled);
  make_boundary_sdm(Udata, FACE_XMAX, mhd_enabled);
  make_boundary_sdm(Udata, FACE_YMIN, mhd_enabled);
  make_boundary_sdm(Udata, FACE_YMAX, mhd_enabled);

  if (dim==3) {

    make_boundary_sdm(Udata, FACE_ZMIN, mhd_enabled);
    make_boundary_sdm(Udata, FACE_ZMAX, mhd_enabled);

  }
    
} // SolverHydroSDM<dim,N>::make_boundaries_sdm_serial
    
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

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor<dim,N> functor(params, sdm_geom, blastParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

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

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);
    
  HydroState2d U0, U1, U2, U3;
  ppkMHD::getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  ppkMHD::primToCons_2D(U0, params.settings.gamma0);
  ppkMHD::primToCons_2D(U1, params.settings.gamma0);
  ppkMHD::primToCons_2D(U2, params.settings.gamma0);
  ppkMHD::primToCons_2D(U3, params.settings.gamma0);
  
  InitFourQuadrantFunctor<dim,N> functor(params, sdm_geom,
					 Udata,
					 U0, U1, U2, U3,
					 xt, yt);
  Kokkos::parallel_for(nbCells, functor);
    
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

  WedgeParams wparams(configMap, 0.0);
  
  InitWedgeFunctor<dim,N> functor(params, sdm_geom, wparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
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

  IsentropicVortexParams iparams(configMap);

  InitIsentropicVortexFunctor<dim,N> functor(params, sdm_geom, iparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::save_solution_impl()
{

  timers[TIMER_IO]->start();

  save_data(U,  Uhost, m_times_saved, m_t);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroSDM::save_solution_impl()

} // namespace sdm

#endif // SOLVER_HYDRO_SDM_H_
