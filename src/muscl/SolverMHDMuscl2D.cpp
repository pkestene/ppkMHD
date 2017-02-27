#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "SolverMHDMuscl2D.h"
#include "HydroParams.h"

// the actual computational functors called in SolverMHDMuscl2D
#include "MHDRunFunctors2D.h"
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
// ==== CLASS SolverMHDMuscl2D IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverMHDMuscl2D::SolverMHDMuscl2D(HydroParams& params, ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Qm_x(), Qm_y(),
  Qp_x(), Qp_y(),
  QEdge_RT(), QEdge_RB(),
  QEdge_LT(), QEdge_LB(),
  Fluxes_x(), Fluxes_y(),
  Emf(),
  isize(params.isize),
  jsize(params.jsize),
  ijsize(params.isize*params.jsize)
{

  m_nCells = ijsize;

  int nbvar = params.nbvar;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  U     = DataArray("U", isize,jsize, nbvar);
  Uhost = Kokkos::create_mirror_view(U);
  U2    = DataArray("U2",isize,jsize, nbvar);
  Q     = DataArray("Q", isize,jsize, nbvar);

  if (params.implementationVersion == 0) {
  
    Qm_x = DataArray("Qm_x", isize,jsize, nbvar);
    Qm_y = DataArray("Qm_y", isize,jsize, nbvar);
    Qp_x = DataArray("Qp_x", isize,jsize, nbvar);
    Qp_y = DataArray("Qp_y", isize,jsize, nbvar);

    QEdge_RT = DataArray("QEdge_RT", isize,jsize, nbvar);
    QEdge_RB = DataArray("QEdge_RB", isize,jsize, nbvar);
    QEdge_LT = DataArray("QEdge_LT", isize,jsize, nbvar);
    QEdge_LB = DataArray("QEdge_LB", isize,jsize, nbvar);
    
    Fluxes_x = DataArray("Fluxes_x", isize,jsize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize,jsize, nbvar);

    Emf = DataArrayScalar("Emf", isize,jsize);
    
  }
  
  // default riemann solver
  // riemann_solver_fn = &SolverMHDMuscl2D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &SolverMHDMuscl2D::riemann_hllc;

  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("blast") ) {

    init_blast(U);
    
  } else if ( !m_problem_name.compare("orszag_tang") ) {
    
    init_orszag_tang(U);
    
  } else {
    
    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - orszag_tang" << std::endl;
    init_orszag_tang(U);

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

} // SolverMHDMuscl2D::SolverMHDMuscl2D


// =======================================================
// =======================================================
/**
 *
 */
SolverMHDMuscl2D::~SolverMHDMuscl2D()
{

} // SolverMHDMuscl2D::~SolverMHDMuscl2D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverMHDMuscl2D::compute_dt_local()
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
  ComputeDtFunctor2D_MHD computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(ijsize, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverMHDMuscl2D::compute_dt_local

// =======================================================
// =======================================================
void SolverMHDMuscl2D::next_iteration_impl()
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
  
} // SolverMHDMuscl2D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverMHDMuscl2D::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverMHDMuscl2D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void SolverMHDMuscl2D::godunov_unsplit_cpu(DataArray data_in, 
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
    
    // trace computation: fill arrays qm_x, qm_y, qp_x, qp_y
    computeTrace(data_in, dt);

    // Compute flux via Riemann solver and update (time integration)
    computeFluxesAndStore(dt);

    // Compute Emf
    computeEmfAndStore(dt);
    
    // actual update with fluxes
    {
      UpdateFunctor2D_MHD functor(params, data_out,
				  Fluxes_x, Fluxes_y, dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
    // actual update with emf
    {
      UpdateEmfFunctor2D functor(params, data_out,
				 Emf, dtdx, dtdy);
      Kokkos::parallel_for(ijsize, functor);
    }
    
  }
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverMHDMuscl2D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor2D_MHD convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijsize, convertToPrimitivesFunctor);
  
} // SolverMHDMuscl2D::convertToPrimitives

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Compute trace (only used in implementation version 2), i.e.
// fill global array qm_x, qmy, qp_x, qp_y
// ///////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeTrace(DataArray Udata, real_t dt)
{

  // local variables
  real_t dtdx;
  real_t dtdy;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;

  // call device functor
  ComputeTraceFunctor2D_MHD computeTraceFunctor(params, Udata, Q,
						Qm_x, Qm_y,
						Qp_x, Qp_y,
						QEdge_RT, QEdge_RB,
						QEdge_LT, QEdge_LB,
						dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeTraceFunctor);
  
} // SolverMHDMuscl2D::computeTrace

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute flux via Riemann solver and store
// //////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeFluxesAndStore(real_t dt)
{
   
  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeFluxesAndStoreFunctor2D_MHD
    computeFluxesAndStoreFunctor(params,
				 Qm_x, Qm_y,
				 Qp_x, Qp_y,
				 Fluxes_x, Fluxes_y,
				 dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeFluxesAndStoreFunctor);
  
} // computeFluxesAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////////////
// Compute EMF via 2D Riemann solver and store
// //////////////////////////////////////////////////////////////////
void SolverMHDMuscl2D::computeEmfAndStore(real_t dt)
{
   
  real_t dtdx = dt / params.dx;
  real_t dtdy = dt / params.dy;

  // call device functor
  ComputeEmfAndStoreFunctor2D
    computeEmfAndStoreFunctor(params,
			      QEdge_RT, QEdge_RB,
			      QEdge_LT, QEdge_LB,
			      Emf,
			      dtdx, dtdy);
  Kokkos::parallel_for(ijsize, computeEmfAndStoreFunctor);
  
} // computeEmfAndStore

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverMHDMuscl2D::make_boundaries(DataArray Udata)
{
  const int ghostWidth=params.ghostWidth;
  int nbIter = ghostWidth*std::max(isize,jsize);
  
  // call device functor
  {
    MakeBoundariesFunctor2D_MHD<FACE_XMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor2D_MHD<FACE_XMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }

  {
    MakeBoundariesFunctor2D_MHD<FACE_YMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor2D_MHD<FACE_YMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  
  
} // SolverMHDMuscl2D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
// void SolverMHDMuscl2D::init_implode(DataArray Udata)
// {

//   InitImplodeFunctor functor(params, Udata);
//   Kokkos::parallel_for(ijsize, functor);
  
// } // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void SolverMHDMuscl2D::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor2D_MHD functor(params, blastParams, Udata);
  Kokkos::parallel_for(ijsize, functor);
  
} // SolverMHDMuscl2D::init_blast

// =======================================================
// =======================================================
/**
 * Orszag-Tang vortex test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
 */
void SolverMHDMuscl2D::init_orszag_tang(DataArray Udata)
{
  
  // init all vars but energy
  {
    InitOrszagTangFunctor2D<INIT_ALL_VAR_BUT_ENERGY> functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }

  // init energy
  {
    InitOrszagTangFunctor2D<INIT_ENERGY> functor(params, Udata);
    Kokkos::parallel_for(ijsize, functor);
  }  
  
} // init_orszag_tang

// =======================================================
// =======================================================
void SolverMHDMuscl2D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverMHDMuscl2D::save_solution_impl()

} // namespace ppkMHD
