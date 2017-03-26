/**
 *
 */
#ifndef SOLVER_HYDRO_MOOD_H_
#define SOLVER_HYDRO_MOOD_H_

#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

namespace mood {

/**
 * Main hydrodynamics data structure.
 */
template<unsigned int dim, unsigned int degree>
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
  DataArray[] Polynomial
  
  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray     U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes
  DataArray Fluxes_x, Fluxes_y;

  //! mood detection
  DataArrayScalar MoodFlags;

  /*
   * MOOD config
   */
  int polynomial_degree;
  STENCIL_ID stencilID;
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
  static const int ncoefs =  mood::binomial<dim+order,dim>();

}; // class SolverHydroMood

} // namespace ppkMHD

#endif // SOLVER_HYDRO_MOOD_H_
