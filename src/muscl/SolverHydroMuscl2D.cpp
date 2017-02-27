#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "SolverHydroMuscl2D.h"
#include "HydroParams.h"

// the actual computational functors called in HydroRun
#include "HydroRunFunctors2D.h"
#include "BoundariesFunctors.h"

// Kokkos
#include "kokkos_shared.h"

// for IO
#include <io/IO_Writer.h>

// for init condition
#include "BlastParams.h"


namespace ppkMHD {

using namespace muscl;

// =======================================================
// ==== CLASS SolverHydroMuscl2D IMPL ====================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl2D::SolverHydroMuscl2D(HydroParams& params,
				       ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(),
  Slopes_x(), Slopes_y(),
  isize(params.isize),
  jsize(params.jsize),
  ijsize(params.isize*params.jsize)
{

  m_nCells = ijsize;

  int nbvar = params.nbvar;
  
  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", isize, jsize, nbvar);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",isize, jsize, nbvar);
  Q     = DataArray("Q", isize, jsize, nbvar);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);
    
  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
    Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = Fluxes_x;
    
  } 
  
  // default riemann solver
  // riemann_solver_fn = &SolverHydroMuscl2D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &SolverHydroMuscl2D::riemann_hllc;
  
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
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(U);

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

} // SolverHydroMuscl2D::SolverHydroMuscl2D


// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl2D::~SolverHydroMuscl2D()
{

} // SolverHydroMuscl2D::~SolverHydroMuscl2D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl2D::compute_dt_local()
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

} // SolverHydroMuscl2D::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMuscl2D::next_iteration_impl()
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
  godunov_unsplit(m_dt);
  
} // SolverHydroMuscl2D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl2D::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverHydroMuscl2D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void SolverHydroMuscl2D::godunov_unsplit_cpu(DataArray data_in, 
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

  if (params.implementationVersion == 0) {
    
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
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor2D computeSlopesFunctor(params, Q, Slopes_x, Slopes_y);
    Kokkos::parallel_for(ijsize, computeSlopesFunctor);

    // now trace along X axis
    {
      ComputeTraceAndFluxes_Functor2D<XDIR> functor(params, Q,
						    Slopes_x, Slopes_y,
						    Fluxes_x,
						    dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // and update along X axis
    {
      UpdateDirFunctor2D<XDIR> functor(params, data_out, Fluxes_x);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // now trace along Y axis
    {
      ComputeTraceAndFluxes_Functor2D<YDIR> functor(params, Q,
						    Slopes_x, Slopes_y,
						    Fluxes_y,
						    dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // and update along Y axis
    {
      UpdateDirFunctor2D<YDIR> functor(params, data_out, Fluxes_y);
      Kokkos::parallel_for(ijsize, functor);
    }
    
  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl2D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl2D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor2D convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  
} // SolverHydroMuscl2D::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMuscl2D::make_boundaries(DataArray Udata)
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
  
} // SolverHydroMuscl2D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void SolverHydroMuscl2D::init_implode(DataArray Udata)
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
void SolverHydroMuscl2D::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor2D functor(params, blastParams, Udata);
  Kokkos::parallel_for(ijsize, functor);

} // SolverHydroMuscl2D::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
void SolverHydroMuscl2D::init_four_quadrant(DataArray Udata)
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
void SolverHydroMuscl2D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl2D::save_solution_impl()

} // namespace ppkMHD
