#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "mood/SolverHydroMood2D.h"
#include "shared/HydroParams.h"

// the actual computational functors called in HydroRun
//#include "mood/HydroRunFunctors2D.h"

#include "shared/BoundariesFunctors.h"

// Kokkos
#include "shared/kokkos_shared.h"

// for IO
#include <utils/io/IO_Writer.h>

// for init condition
#include "shared/BlastParams.h"


namespace ppkMHD {

using namespace mood;

// =======================================================
// ==== CLASS SolverHydroMood2D IMPL ====================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMood2D::SolverHydroMood2D(HydroParams& params,
				     ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), U2(),
  Fluxes_x(), Fluxes_y(),
  isize(params.isize),
  jsize(params.jsize),
  ijsize(params.isize*params.jsize),
  polynomial_degree(1),
  stencilID(STENCIL_2D_ORDER1)
{

  m_nCells = ijsize;

  int nbvar = params.nbvar;
  
  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", isize, jsize, nbvar);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",isize, jsize, nbvar);

  Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
  Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);

  /* MOOD configuration */
  polynomial_degree = configMap.getInteger("mood","polynomial_degree",0);
  std::string stencilId_str = configMap.getString("mood", "stencil_id", "STENCIL_2D_DEGREE1");
  stencilID = StencilUtils::get_stencil_from_string(stencilId_str);

  stencil = Stencil(StencilId);
  
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

} // SolverHydroMood2D::SolverHydroMood2D


// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMood2D::~SolverHydroMood2D()
{

} // SolverHydroMood2D::~SolverHydroMood2D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMood2D::compute_dt_local()
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
  ComputeDtFunctor2D computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(ijsize, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMood2D::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMood2D::next_iteration_impl()
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
  
} // SolverHydroMood2D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMood2D::time_integration(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    time_integration_impl(U , U2, dt);
  } else {
    time_integration_impl(U2, U , dt);
  }
  
} // SolverHydroMood2D::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of MOOD scheme
// ///////////////////////////////////////////
void SolverHydroMood2D::time_integration_impl(DataArray data_in, 
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

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  // compute fluxes
  {
    ComputeAndStoreFluxesFunctor2D functor(params, Q,
					   Fluxes_x, Fluxes_y,
					   dtdx, dtdy);
    Kokkos::parallel_for(ijsize, functor);
  }
  
  // actual update
  {
    UpdateFunctor2D functor(params, data_out,
			    Fluxes_x, Fluxes_y);
    Kokkos::parallel_for(ijsize, functor);
  }
  
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMood2D::time_integration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMood2D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor2D convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  
} // SolverHydroMood2D::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMood2D::make_boundaries(DataArray Udata)
{
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
  
} // SolverHydroMood2D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void SolverHydroMood2D::init_implode(DataArray Udata)
{

  InitImplodeFunctor2D functor(params, Udata);
  Kokkos::parallel_for(ijsize, functor);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void SolverHydroMood2D::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor2D functor(params, blastParams, Udata);
  Kokkos::parallel_for(ijsize, functor);

} // SolverHydroMood2D::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
void SolverHydroMood2D::init_four_quadrant(DataArray Udata)
{

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);

  HydroState2d U0, U1, U2, U3;
  getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  primToCons_2D(U0, params.settings.gamma0);
  primToCons_2D(U1, params.settings.gamma0);
  primToCons_2D(U2, params.settings.gamma0);
  primToCons_2D(U3, params.settings.gamma0);

  InitFourQuadrantFunctor2D functor(params, Udata, configNumber,
				    U0, U1, U2, U3,
				    xt, yt);
  Kokkos::parallel_for(ijsize, functor);
  
} // init_four_quadrant

// =======================================================
// =======================================================
void SolverHydroMood2D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMoo2D::save_solution_impl()

} // namespace ppkMHD
