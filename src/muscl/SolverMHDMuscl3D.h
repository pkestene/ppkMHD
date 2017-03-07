/**
 *
 */
#ifndef SOLVER_MHD_MUSCL_3D_H_
#define SOLVER_MHD_MUSCL_3D_H_

#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

namespace ppkMHD {

/**
 * Main magnehydrodynamics data structure 3D.
 */
class SolverMHDMuscl3D : public SolverBase
{

public:

  using DataArray     = DataArray3d;
  using DataArrayHost = DataArray3dHost;

  SolverMHDMuscl3D(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverMHDMuscl3D();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverMHDMuscl3D* solver = new SolverMHDMuscl3D(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */
  
  DataArray Qm_x; /*!< hydrodynamics Riemann states array implementation 2 */
  DataArray Qm_y; /*!< hydrodynamics Riemann states array */
  DataArray Qm_z; /*!< hydrodynamics Riemann states array */

  DataArray Qp_x; /*!< hydrodynamics Riemann states array */
  DataArray Qp_y; /*!< hydrodynamics Riemann states array */
  DataArray Qp_z; /*!< hydrodynamics Riemann states array */
  
  DataArray QEdge_RT;
  DataArray QEdge_RB;
  DataArray QEdge_LT;
  DataArray QEdge_LB;

  DataArray QEdge_RT2;
  DataArray QEdge_RB2;
  DataArray QEdge_LT2;
  DataArray QEdge_LB2;

  DataArray QEdge_RT3;
  DataArray QEdge_RB3;
  DataArray QEdge_LT3;
  DataArray QEdge_LB3;

  DataArray Fluxes_x;
  DataArray Fluxes_y;
  DataArray Fluxes_z;

  DataArrayVector3 Emf; /*!< electromotive forces */

  DataArrayVector3 ElecField;

  DataArrayVector3 DeltaA;
  DataArrayVector3 DeltaB;
  DataArrayVector3 DeltaC;
  
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

  void computeElectricField(DataArray Udata);
  void computeMagSlopes(DataArray Udata);
  
  void computeTrace(DataArray Udata, real_t dt);
  
  void computeFluxesAndStore(real_t dt);
  void computeEmfAndStore(real_t dt);
  
  void make_boundaries(DataArray Udata);

  // host routines (initialization)
  //void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_orszag_tang(DataArray Udata);

  void save_solution_impl();

  // host routines (save data to file, device data are copied into host
  // inside this routine)
  void saveVTK(DataArray Udata, int iStep, std::string name);
  
  int isize, jsize, ksize, ijsize, ijksize;
  
}; // class SolverMHDMuscl3D

} // namespace ppkMHD

#endif // SOLVER_MHD_MUSCL_3D_H_
