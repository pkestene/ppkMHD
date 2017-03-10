/**
 *
 */
#ifndef SOLVER_HYDRO_MUSCL_3D_H_
#define SOLVER_HYDRO_MUSCL_3D_H_

#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

namespace ppkMHD {

/**
 * Main hydrodynamics data structure for 3D MUSCL-Hancock scheme.
 */
class SolverHydroMuscl3D : public SolverBase
{

public:

  using DataArray     = DataArray3d;
  using DataArrayHost = DataArray3dHost;
  
  SolverHydroMuscl3D(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroMuscl3D();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroMuscl3D* solver = new SolverHydroMuscl3D(params, configMap);

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
  int ijsize, ijksize;
  
}; // class SolverHydroMuscl3D

} // namespace ppkMHD

#endif // SOLVER_HYDRO_MUSCL_3D_H_
