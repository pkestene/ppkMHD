#include <string> 
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "SolverHydroMuscl3D.h"
#include "shared/HydroParams.h"

// the actual computational functors called in HydroRun
#include "HydroRunFunctors3D.h"

// Init conditions functors
#include "muscl/HydroInitFunctors3D.h"

// border conditions functors
#include "shared/BoundariesFunctors.h"

// Kokkos
#include "shared/kokkos_shared.h"

// for IO
#include <utils/io/IO_Writer.h>

// for init condition
#include "shared/BlastParams.h"

namespace ppkMHD {

using namespace muscl;

// =======================================================
// ==== CLASS SolverHydroMuscl3D IMPL ====================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl3D::SolverHydroMuscl3D(HydroParams& params, ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  ijsize(params.isize*params.jsize),
  ijksize(params.isize*params.jsize*params.ksize)
{

  m_nCells = ijksize;

  int nbvar = params.nbvar;
 
  /*
   * memory allocation (use sizes with ghosts included)
   *
   * Note that Uhost is not just a view to U, Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize Uhost.
   */
  U     = DataArray("U", isize,jsize,ksize, nbvar);
  Uhost = Kokkos::create_mirror(U);
  U2    = DataArray("U2",isize,jsize,ksize, nbvar);
  Q     = DataArray("Q", isize,jsize,ksize, nbvar);

  if (params.implementationVersion == 0) {

    Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize,jsize,ksize, nbvar);
    Fluxes_z = DataArray("Fluxes_z", isize,jsize,ksize, nbvar);
    
  } else if (params.implementationVersion == 1) {

    Slopes_x = DataArray("Slope_x", isize,jsize,ksize, nbvar);
    Slopes_y = DataArray("Slope_y", isize,jsize,ksize, nbvar);
    Slopes_z = DataArray("Slope_z", isize,jsize,ksize, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
    Fluxes_y = Fluxes_x;
    Fluxes_z = Fluxes_x;
    
  }
  
  // default riemann solver
  // riemann_solver_fn = &SolverHydroMuscl3D::riemann_approx;
  // if (!riemannSolverStr.compare("hllc"))
  //   riemann_solver_fn = &SolverHydroMuscl3D::riemann_hllc;
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(U);

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

} // SolverHydroMuscl3D::SolverHydroMuscl3D


// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl3D::~SolverHydroMuscl3D()
{

} // SolverHydroMuscl3D::~SolverHydroMuscl3D

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl3D::compute_dt_local()
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
  ComputeDtFunctor3D computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(ijksize, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMuscl3D::compute_dt

// =======================================================
// =======================================================
void SolverHydroMuscl3D::next_iteration_impl()
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
  
} // SolverHydroMuscl3D::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl3D::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverHydroMuscl3D::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
void SolverHydroMuscl3D::godunov_unsplit_cpu(DataArray data_in, 
					     DataArray data_out, 
					     real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;
  
  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

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
      ComputeAndStoreFluxesFunctor3D functor(params, Q,
					     Fluxes_x, Fluxes_y, Fluxes_z,
					     dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }

    // actual update
    {
      UpdateFunctor3D functor(params, data_out,
			      Fluxes_x, Fluxes_y, Fluxes_z);
      Kokkos::parallel_for(ijksize, functor);
    }
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor3D computeSlopesFunctor(params, Q,
						Slopes_x, Slopes_y, Slopes_z);
    Kokkos::parallel_for(ijksize, computeSlopesFunctor);

    // now trace along X axis
    {
      ComputeTraceAndFluxes_Functor3D<XDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_x,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along X axis
    {
      UpdateDirFunctor3D<XDIR> functor(params, data_out, Fluxes_x);
      Kokkos::parallel_for(ijksize, functor);
    }

    // now trace along Y axis
    {
      ComputeTraceAndFluxes_Functor3D<YDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_y,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along Y axis
    {
      UpdateDirFunctor3D<YDIR> functor(params, data_out, Fluxes_y);
      Kokkos::parallel_for(ijksize, functor);
    }

    // now trace along Z axis
    {
      ComputeTraceAndFluxes_Functor3D<ZDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_z,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(ijksize, functor);
    }
    
    // and update along Z axis
    {
      UpdateDirFunctor3D<ZDIR> functor(params, data_out, Fluxes_z);
      Kokkos::parallel_for(ijksize, functor);
    }

  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl3D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl3D::convertToPrimitives(DataArray Udata)
{

  // call device functor
  ConvertToPrimitivesFunctor3D convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(ijksize, convertToPrimitivesFunctor);
  
} // SolverHydroMuscl3D::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMuscl3D::make_boundaries(DataArray Udata)
{
  const int ghostWidth=params.ghostWidth;

  int max_size = std::max(isize,jsize);
  max_size = std::max(max_size,ksize);
  int nbIter = ghostWidth * max_size * max_size;
  
  // call device functor
  {
    MakeBoundariesFunctor3D<FACE_XMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }  
  {
    MakeBoundariesFunctor3D<FACE_XMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }

  {
    MakeBoundariesFunctor3D<FACE_YMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_YMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }

  {
    MakeBoundariesFunctor3D<FACE_ZMIN> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  {
    MakeBoundariesFunctor3D<FACE_ZMAX> functor(params, Udata);
    Kokkos::parallel_for(nbIter, functor);
  }
  
} // SolverHydroMuscl3D::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void SolverHydroMuscl3D::init_implode(DataArray Udata)
{

  InitImplodeFunctor3D functor(params, Udata);
  Kokkos::parallel_for(ijksize, functor);

} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void SolverHydroMuscl3D::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);

  InitBlastFunctor3D functor(params, blastParams, Udata);
  Kokkos::parallel_for(ijksize, functor);

} // SolverHydroMuscl3D::init_blast

// =======================================================
// =======================================================
void SolverHydroMuscl3D::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl3D::save_solution_impl()

} // namespace ppkMHD
