/**
 *
 */
#ifndef SOLVER_HYDRO_MOOD_H_
#define SOLVER_HYDRO_MOOD_H_

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
#include "shared/initRiemannConfig2d.h"

// mood
#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"
#include "mood/Polynomial.h"
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

#include "mood/MoodFunctors.h"
#include "mood/MoodInitFunctors.h"
#include "mood/mood_utils.h"

// for IO
#include <utils/io/IO_Writer.h>

// for init condition
#include "shared/BlastParams.h"

namespace mood {

/**
 * Main hydrodynamics data structure.
 */
template<int dim, int degree>
class SolverHydroMood : public ppkMHD::SolverBase
{

public:

  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim==2,DataArray2dHost,DataArray3dHost>::type;

  
  SolverHydroMood(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroMood();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroMood<dim,degree>* solver = new SolverHydroMood<dim,degree>(params, configMap);

    return solver;
  }
  
  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */

  //! reconstructing polynomial
  //DataArray[] Polynomial
  
  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray     U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes
  DataArray Fluxes_x, Fluxes_y, Fluxes_z;

  //! mood detection
  DataArrayScalar MoodFlags;

  /*
   * MOOD config
   */
  STENCIL_ID stencilId;
  Stencil stencil;
  Matrix geomMatrix;
  
  /*
   * methods
   */

  //! compute time step inside an MPI process, at shared memory level.
  double compute_dt_local();

  //! perform 1 time step (time integration).
  void next_iteration_impl();

  //! numerical scheme
  void time_integration(real_t dt);
  
  void time_integration_impl(DataArray data_in, 
			     DataArray data_out, 
			     real_t dt);
  
  void make_boundaries(DataArray Udata);

  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_four_quadrant(DataArray Udata);

  void save_solution_impl();
  
  int isize, jsize, ksize, nbCells;

  //! total number of coefficients in the polynomial
  static const int ncoefs =  mood::binomial<dim+degree,dim>();

}; // class SolverHydroMood


// =======================================================
// ==== CLASS SolverHydroMood IMPL =======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int degree>
SolverHydroMood<dim,degree>::SolverHydroMood(HydroParams& params,
					     ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), U2(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCells(params.isize*params.jsize*params.ksize),
  stencilId(select_stencil(dim,degree)),
  stencil(stencilId)
{

  m_nCells = nbCells;

  int nbvar = params.nbvar;
  
  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (dim==2) {

    U     = DataArray("U", isize, jsize, nbvar);
    Uhost = Kokkos::create_mirror_view(U);
    U2    = DataArray("U2",isize, jsize, nbvar);
    
    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);
    
  } else if (dim==3) {

    U     = DataArray("U", isize, jsize, ksize, nbvar);
    Uhost = Kokkos::create_mirror_view(U);
    U2    = DataArray("U2",isize, jsize, ksize, nbvar);
    
    Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, ksize, nbvar);
    Fluxes_z = DataArray("Fluxes_z", isize, jsize, ksize, nbvar);

  }
    
  /* MOOD configuration */
  //stencilId = select_stencil(dim,degree);

  //stencil = Stencil(stencilId);
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(U);

  } else if ( !m_problem_name.compare("four_quadrant") ) {

    init_four_quadrant(U);

  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - Four Quadrant" << std::endl;
    init_four_quadrant(U);

  }
  std::cout << "##########################" << "\n";
  std::cout << "Solver is " << m_solver_name << "\n";
  std::cout << "Problem (init condition) is " << m_problem_name << "\n";
  std::cout << "##########################" << "\n";

  // print parameters on screen
  params.print();
  std::cout << "##########################" << "\n";

  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);

} // SolverHydroMood::SolverHydroMood

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int degree>
SolverHydroMood<dim,degree>::~SolverHydroMood()
{

} // SolverHydroMood::~SolverHydroMood


// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template<int dim, int degree>
double SolverHydroMood<dim,degree>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  // call device functor
  //ComputeDtFunctor computeDtFunctor(params, Udata);
  //Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMood::compute_dt_local

// =======================================================
// =======================================================
template<int dim, int degree>
void SolverHydroMood<dim,degree>::next_iteration_impl()
{

  if (m_iteration % 10 == 0) {
    std::cout << "time step=" << m_iteration << std::endl;
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
  
} // SolverHydroMood::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template<int dim, int degree>
void SolverHydroMood<dim,degree>::time_integration(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    time_integration_impl(U , U2, dt);
  } else {
    time_integration_impl(U2, U , dt);
  }
  
} // SolverHydroMood::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of MOOD scheme
// ///////////////////////////////////////////
template<int dim, int degree>
void SolverHydroMood<dim,degree>::time_integration_impl(DataArray data_in, 
							DataArray data_out, 
							real_t dt)
{
  
  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  // compute fluxes
  // {
  //   ComputeAndStoreFluxesFunctor functor(params, Q,
  // 					   Fluxes_x, Fluxes_y,
  // 					   dtdx, dtdy);
  //   Kokkos::parallel_for(nbCells, functor);
  // }
  
  // actual update
  // {
  //   UpdateFunctor functor(params, data_out,
  // 			    Fluxes_x, Fluxes_y);
  //   Kokkos::parallel_for(nbCells, functor);
  // }
  
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMood::time_integration_impl


// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim, int degree>
void SolverHydroMood<dim,degree>::make_boundaries(DataArray Udata)
{

  if (dim==2) {

    const int ghostWidth=params.ghostWidth;
    int nbIter = ghostWidth*std::max(isize,jsize);
    
    // call device functor
    {
      MakeBoundariesFunctor2D<FACE_XMIN> functor(params, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    {
      MakeBoundariesFunctor2D<FACE_XMAX> functor(params, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    
    {
      MakeBoundariesFunctor2D<FACE_YMIN> functor(params, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    {
      MakeBoundariesFunctor2D<FACE_YMAX> functor(params, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    
  } else if (dim==3) {

    // const int ghostWidth=params.ghostWidth;
    
    // int max_size = std::max(isize,jsize);
    // max_size = std::max(max_size,ksize);
    // int nbIter = ghostWidth * max_size * max_size;
    
    // // call device functor
    // {
    //   MakeBoundariesFunctor3D<FACE_XMIN> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }  
    // {
    //   MakeBoundariesFunctor3D<FACE_XMAX> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }
    
    // {
    //   MakeBoundariesFunctor3D<FACE_YMIN> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }
    // {
    //   MakeBoundariesFunctor3D<FACE_YMAX> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }
    
    // {
    //   MakeBoundariesFunctor3D<FACE_ZMIN> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }
    // {
    //   MakeBoundariesFunctor3D<FACE_ZMAX> functor(params, Udata);
    //   Kokkos::parallel_for(nbIter, functor);
    // }

  } // end 3D
  
} // SolverHydroMood::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template<int dim, int degree>
void SolverHydroMood<dim,degree>::init_implode(DataArray Udata)
{

  InitImplodeFunctor<dim,degree> functor(params, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template<int dim, int degree>
void SolverHydroMood<dim,degree>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor<dim,degree> functor(params, blastParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroMood::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant  riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template<int dim, int degree>
void SolverHydroMood<dim,degree>::init_four_quadrant(DataArray Udata)
{

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);

  if (dim==2) {
    
    HydroState2d U0, U1, U2, U3;
    ppkMHD::getRiemannConfig2d(configNumber, U0, U1, U2, U3);
    
    ppkMHD::primToCons_2D(U0, params.settings.gamma0);
    ppkMHD::primToCons_2D(U1, params.settings.gamma0);
    ppkMHD::primToCons_2D(U2, params.settings.gamma0);
    ppkMHD::primToCons_2D(U3, params.settings.gamma0);
    
    InitFourQuadrantFunctor<dim,degree> functor(params, Udata,
						U0, U1, U2, U3,
						xt, yt);
    Kokkos::parallel_for(nbCells, functor);
    
  } else if (dim==3) {

    // TODO - TODO - TODO
    
  }
  
} // init_four_quadrant

// =======================================================
// =======================================================
template<int dim, int degree>
void SolverHydroMood<dim,degree>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMood::save_solution_impl()

} // namespace mood

#endif // SOLVER_HYDRO_MOOD_H_
