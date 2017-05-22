/**
 * Class SolverHydroMuscl implementation.
 *
 * Main class for solving hydrodynamics (Euler) with MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_HYDRO_MUSCL_H_
#define SOLVER_HYDRO_MUSCL_H_

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

// the actual computational functors called in HydroRun
#include "muscl/HydroRunFunctors2D.h"
#include "muscl/HydroRunFunctors3D.h"

// Init conditions functors
#include "muscl/HydroInitFunctors2D.h"
#include "muscl/HydroInitFunctors3D.h"

// border conditions functors
#include "shared/BoundariesFunctors.h"

// for IO
#include <utils/io/IO_Writer.h>

// for init condition
#include "shared/BlastParams.h"
#include "shared/IsentropicVortexParams.h"

namespace ppkMHD { namespace muscl {

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
template<int dim>
class SolverHydroMuscl : public ppkMHD::SolverBase
{

public:

  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim==2,DataArray2dHost,DataArray3dHost>::type;

  SolverHydroMuscl(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroMuscl();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroMuscl<dim>* solver = new SolverHydroMuscl<dim>(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */

  /* implementation 0 */
  DataArray Fluxes_x; /*!< implementation 0 */
  DataArray Fluxes_y; /*!< implementation 0 */
  DataArray Fluxes_z; /*!< implementation 0 */
  
  /* implementation 1 only */
  DataArray Slopes_x; /*!< implementation 1 only */
  DataArray Slopes_y; /*!< implementation 1 only */
  DataArray Slopes_z; /*!< implementation 1 only */


  //riemann_solver_t riemann_solver_fn; /*!< riemann solver function pointer */

  /*
   * methods
   */

  //! compute time step inside an MPI process, at shared memory level.
  double compute_dt_local();

  //! perform 1 time step (time integration).
  void next_iteration_impl();

  //! numerical scheme
  void godunov_unsplit(real_t dt);
  
  void godunov_unsplit_cpu(DataArray data_in, 
			   DataArray data_out, 
			   real_t dt);
  
  void convertToPrimitives(DataArray Udata);
  
  //void computeTrace(DataArray Udata, real_t dt);
  
  void computeFluxesAndUpdate(DataArray Udata, 
			      real_t dt);

  // fill boundaries / ghost 2d
  template<int dim_ = dim>
  void make_boundaries(typename std::enable_if<dim_==2,DataArray2d>::type Udata);

  // fill boundaries / ghost 3d
  template<int dim_ = dim>
  void make_boundaries(typename std::enable_if<dim_==3,DataArray3d>::type Udata);

  // host routines (initialization)
  template<int dim_ = dim>
  void init(typename std::enable_if<dim_==2,DataArray2d>::type Udata);
  template<int dim_ = dim>
  void init(typename std::enable_if<dim_==3,DataArray3d>::type Udata);
  
  void init_implode(DataArray Udata); // 2d and 3d
  void init_blast(DataArray Udata); // 2d and 3d
  void init_four_quadrant(DataArray Udata); // 2d only
  void init_isentropic_vortex(DataArray Udata); // 2d only

  // output
  void save_solution_impl();
  
  int isize, jsize, ksize;
  int nbCells;
  
}; // class SolverHydroMuscl

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template<int dim>
SolverHydroMuscl<dim>::SolverHydroMuscl(HydroParams& params,
					ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCells(params.isize*params.jsize)
{

  if (dim==3)
    nbCells = params.isize*params.jsize*params.ksize;
  
  m_nCells = nbCells;
  
  int nbvar = params.nbvar;
 
  long long int total_mem_size = 0;

  /*
   * memory allocation (use sizes with ghosts included)
   *
   * Note that Uhost is not just a view to U, Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize Uhost.
   */
  if (dim==2) {

    U     = DataArray("U", isize, jsize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2    = DataArray("U2",isize, jsize, nbvar);
    Q     = DataArray("Q", isize, jsize, nbvar);

    total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 3;// 1+1+1 for U+U2+Q
    
    if (params.implementationVersion == 0) {
      
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);
      
      total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 2;// 1+1 for Fluxes_x+Fluxes_y

    } else if (params.implementationVersion == 1) {
      
      Slopes_x = DataArray("Slope_x", isize, jsize, nbvar);
      Slopes_y = DataArray("Slope_y", isize, jsize, nbvar);
      
      // direction splitting (only need one flux array)
      Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
      Fluxes_y = Fluxes_x;
      
      total_mem_size += isize*jsize*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Fluxes_x

    } 

  } else {

    U     = DataArray("U", isize,jsize,ksize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2    = DataArray("U2",isize,jsize,ksize, nbvar);
    Q     = DataArray("Q", isize,jsize,ksize, nbvar);
    
    total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*3;// 1+1+1=3 for U+U2+Q

    if (params.implementationVersion == 0) {
      
      Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
      Fluxes_y = DataArray("Fluxes_y", isize,jsize,ksize, nbvar);
      Fluxes_z = DataArray("Fluxes_z", isize,jsize,ksize, nbvar);
      
      total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*3;// 1+1+1=3 Fluxes

    } else if (params.implementationVersion == 1) {
      
      Slopes_x = DataArray("Slope_x", isize,jsize,ksize, nbvar);
      Slopes_y = DataArray("Slope_y", isize,jsize,ksize, nbvar);
      Slopes_z = DataArray("Slope_z", isize,jsize,ksize, nbvar);
      
      // direction splitting (only need one flux array)
      Fluxes_x = DataArray("Fluxes_x", isize,jsize,ksize, nbvar);
      Fluxes_y = Fluxes_x;
      Fluxes_z = Fluxes_x;
      
      total_mem_size += isize*jsize*ksize*nbvar*sizeof(real_t)*4;// 1+1+1+1=4 Slopes
    }
    
  } // dim == 2 / 3
  
  // perform init condition
  init(U);
  
  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);
  
  // compute initialize time step
  compute_dt();

  std::cout << "##########################" << "\n";
  std::cout << "Solver is " << m_solver_name << "\n";
  std::cout << "Problem (init condition) is " << m_problem_name << "\n";
  std::cout << "##########################" << "\n";

  // print parameters on screen
  params.print();
  std::cout << "##########################" << "\n";
  std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
  std::cout << "##########################" << "\n";

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
template<int dim>
SolverHydroMuscl<dim>::~SolverHydroMuscl()
{

} // SolverHydroMuscl::~SolverHydroMuscl

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template<int dim>
double SolverHydroMuscl<dim>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  // alias to actual device functor
  using ComputeDtFunctor =
    typename std::conditional<dim==2,
			      ComputeDtFunctor2D,
			      ComputeDtFunctor3D>::type;

  // call device functor
  ComputeDtFunctor computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::next_iteration_impl()
{

  if (m_iteration % 10 == 0) {
    //std::cout << "time step=" << m_iteration << std::endl;
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
  godunov_unsplit(m_dt);
  
} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_cpu(U , U2, dt);
  } else {
    godunov_unsplit_cpu(U2, U , dt);
  }
  
} // SolverHydroMuscl::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of Godunov scheme
// ///////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::godunov_unsplit_cpu(DataArray data_in, 
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
      Kokkos::parallel_for(nbCells, functor);
    }

    // actual update
    {
      UpdateFunctor3D functor(params, data_out,
			      Fluxes_x, Fluxes_y, Fluxes_z);
      Kokkos::parallel_for(nbCells, functor);
    }
    
  } else if (params.implementationVersion == 1) {

    // call device functor to compute slopes
    ComputeSlopesFunctor3D computeSlopesFunctor(params, Q,
						Slopes_x, Slopes_y, Slopes_z);
    Kokkos::parallel_for(nbCells, computeSlopesFunctor);

    // now trace along X axis
    {
      ComputeTraceAndFluxes_Functor3D<XDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_x,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(nbCells, functor);
    }
    
    // and update along X axis
    {
      UpdateDirFunctor3D<XDIR> functor(params, data_out, Fluxes_x);
      Kokkos::parallel_for(nbCells, functor);
    }

    // now trace along Y axis
    {
      ComputeTraceAndFluxes_Functor3D<YDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_y,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(nbCells, functor);
    }
    
    // and update along Y axis
    {
      UpdateDirFunctor3D<YDIR> functor(params, data_out, Fluxes_y);
      Kokkos::parallel_for(nbCells, functor);
    }

    // now trace along Z axis
    {
      ComputeTraceAndFluxes_Functor3D<ZDIR> functor(params, Q,
						    Slopes_x, Slopes_y, Slopes_z,
						    Fluxes_z,
						    dtdx, dtdy, dtdz);
      Kokkos::parallel_for(nbCells, functor);
    }
    
    // and update along Z axis
    {
      UpdateDirFunctor3D<ZDIR> functor(params, data_out, Fluxes_z);
      Kokkos::parallel_for(nbCells, functor);
    }

  } // end params.implementationVersion == 1
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl3D::godunov_unsplit_cpu

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
template<int dim>
void SolverHydroMuscl<dim>::convertToPrimitives(DataArray Udata)
{

  // alias to actual device functor
  using ConvertToPrimitivesFunctor =
    typename std::conditional<dim==2,
			      ConvertToPrimitivesFunctor2D,
			      ConvertToPrimitivesFunctor3D>::type;

  // call device functor
  ConvertToPrimitivesFunctor convertToPrimitivesFunctor(params, Udata, Q);
  Kokkos::parallel_for(nbCells, convertToPrimitivesFunctor);
  
} // SolverHydroMuscl::convertToPrimitives

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim>
template<int dim_>
void SolverHydroMuscl<dim>::make_boundaries(typename std::enable_if<dim_==2,DataArray2d>::type Udata)
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
  
} // SolverHydroMuscl::make_boundaries

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim>
template<int dim_>
void SolverHydroMuscl<dim>::make_boundaries(typename std::enable_if<dim_==3,DataArray3d>::type  Udata)
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
  
} // SolverHydroMuscl::make_boundaries

// =======================================================
// =======================================================
template<int dim>
template<int dim_>
void SolverHydroMuscl<dim>::init(typename std::enable_if<dim_==2,DataArray2d>::type Udata)
{

  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(Udata);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(Udata);

  } else if ( !m_problem_name.compare("four_quadrant") and (dim==2)) {

    init_four_quadrant(Udata);

  } else if ( !m_problem_name.compare("isentropic_vortex") and (dim==2)) {

    init_isentropic_vortex(Udata);

  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(Udata);

  }

} // SolverHydroMuscl::init / 2d

// =======================================================
// =======================================================
template<int dim>
template<int dim_>
void SolverHydroMuscl<dim>::init(typename std::enable_if<dim_==3,DataArray3d>::type Udata)
{

  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(Udata);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(Udata);

  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - implode" << std::endl;
    init_implode(Udata);

  }

} // SolverHydroMuscl::init / 3d

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template<int dim>
void SolverHydroMuscl<dim>::init_implode(DataArray Udata)
{

  // alias to actual device functor
  using InitImplodeFunctor =
    typename std::conditional<dim==2,
			      InitImplodeFunctor2D,
			      InitImplodeFunctor3D>::type;

  // perform init
  InitImplodeFunctor functor(params, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroMuscl::init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template<int dim>
void SolverHydroMuscl<dim>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);

  // alias to actual device functor
  using InitBlastFunctor =
    typename std::conditional<dim==2,
			      InitBlastFunctor2D,
			      InitBlastFunctor3D>::type;

  // perform init
  InitBlastFunctor functor(params, blastParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroMuscl::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template<int dim>
void SolverHydroMuscl<dim>::init_four_quadrant(DataArray Udata)
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
  Kokkos::parallel_for(nbCells, functor);
  
} // SolverHydroMuscl::init_four_quadrant

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template<int dim>
void SolverHydroMuscl<dim>::init_isentropic_vortex(DataArray Udata)
{

  IsentropicVortexParams iparams(configMap);
  
  InitIsentropicVortexFunctor2D functor(params, iparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // SolverHydroMuscl::init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim>
void SolverHydroMuscl<dim>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved);
  else
    save_data(U2, Uhost, m_times_saved);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl::save_solution_impl()

} // namespace muscl

} // namespace ppkMHD

#endif // SOLVER_HYDRO_MUSCL_H_
