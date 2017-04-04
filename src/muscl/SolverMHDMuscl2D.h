/**
 *
 */
#ifndef SOLVER_MHD_MUSCL_2D_H_
#define SOLVER_MHD_MUSCL_2D_H_

#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

namespace ppkMHD {

/**
 * Main magnehydrodynamics data structure 2D.
 */
class SolverMHDMuscl2D : public SolverBase
{

public:

  using DataArray     = DataArray2d;
  using DataArrayHost = DataArray2dHost;

  SolverMHDMuscl2D(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverMHDMuscl2D();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverMHDMuscl2D* solver = new SolverMHDMuscl2D(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */
  
  /* implementation 2 only */
  DataArray Qm_x; /*!< hydrodynamics Riemann states array implementation 2 */
  DataArray Qm_y; /*!< hydrodynamics Riemann states array */
  DataArray Qp_x; /*!< hydrodynamics Riemann states array */
  DataArray Qp_y; /*!< hydrodynamics Riemann states array */

  DataArray QEdge_RT;
  DataArray QEdge_RB;
  DataArray QEdge_LT;
  DataArray QEdge_LB;

  DataArray Fluxes_x;
  DataArray Fluxes_y;

  DataArrayScalar Emf; /*!< electromotive force */
  
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
  
  void computeTrace(DataArray Udata, real_t dt);
  
  void computeFluxesAndStore(real_t dt);
  void computeEmfAndStore(real_t dt);
  
  void make_boundaries(DataArray Udata);

  // host routines (initialization)
  //void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_orszag_tang(DataArray Udata);

  void save_solution_impl();
  
  int isize, jsize, ijsize;
  
}; // class SolverMHDMuscl2D

} // namespace ppkMHD

#endif // SOLVER_MHD_MUSCL_2D_H_
