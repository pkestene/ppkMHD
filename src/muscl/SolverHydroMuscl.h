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

namespace muscl {

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
template<inbt dim>
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
  
  void make_boundaries(DataArray Udata);

  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);

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
SolverHydroMuscl::SolverHydroMuscl(HydroParams& params,
				   ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z(),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCels(params.isize*params.jsize)
{

  if (dim==3)
    nbCells = params.isize*params.jsize*params.ksize;
  
  m_nCells = ijksize;
  
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

  } else if ( !m_problem_name.compare("four_quadrant") and (dim==2)) {

    init_four_quadrant(U);

  } else if ( !m_problem_name.compare("isentropic_vortex") and (dim==2)) {

    init_isentropic_vortex(U);

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
  std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
  std::cout << "##########################" << "\n";

  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl::~SolverHydroMuscl()
{

} // SolverHydroMuscl::~SolverHydroMuscl

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl::compute_dt_local()
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
  using ComputeDtFunctor =
    typename std::conditional<dim==2,
			      ComputeDtFunctor2d<degree>,
			      ComputeDtFunctor3d<degree>>::type;

  // call device functor
  ComputeDtFunctor computeDtFunctor(params, Udata);
  Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMuscl::next_iteration_impl()
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


} // namespace muscl

#endif // SOLVER_HYDRO_MUSCL_H_
