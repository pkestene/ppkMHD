/**
 * Main class to drive a Spectral Difference Method numerical scheme.
 */
#ifndef SOLVER_HYDRO_SDM_H_
#define SOLVER_HYDRO_SDM_H_

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
#include "shared/mpiBorderUtils.h"
//#include "shared/BoundariesFunctors.h"
//#include "shared/BoundariesFunctorsWedge.h"
#include "shared/initRiemannConfig2d.h"
#include "shared/EulerEquations.h"

// sdm
#include "sdm/SDM_Geometry.h"

// sdm functors (where the action takes place)
#include "sdm/HydroInitFunctors.h"
#include "sdm/SDM_Dt_Functor.h"
#include "sdm/SDM_Interpolate_Functors.h"
#include "sdm/SDM_Interpolate_viscous_Functors.h"
#include "sdm/SDM_Flux_Functors.h"
#include "sdm/SDM_Flux_with_Limiter_Functors.h"
#include "sdm/SDM_Run_Functors.h"
#include "sdm/SDM_Boundaries_Functors.h"
#include "sdm/SDM_Boundaries_Functors_Wedge.h"
#include "sdm/SDM_Boundaries_Functors_Jet.h"
#include "sdm/SDM_Limiter_Functors.h"
#include "sdm/SDM_Positivity_preserving.h"

// for IO
#include "utils/io/IO_Writer_SDM.h"

// for specific init / border conditions
#include "shared/BlastParams.h"
#include "shared/KHParams.h"
#include "shared/WedgeParams.h"
#include "shared/JetParams.h"

namespace sdm {

/**
 * Main hydrodynamics data structure driving SDM numerical solver.
 *
 * \tparam dim dimension of the domain (2 or 3)
 * \tparam N is the number of solution points per direction
 *
 * The total number of solution points per cells is N^2 in 2D, and N^3 in 3D.
 * We use row-major format to enumerate DoF (degrees of freedom) inside cell,
 * i.e. we will define a mapping  
 * for example in 2D:
 *    index = DofMap(i,j,iv) computed as i+N*j+N*N*iv
 * DofMap returns index where one can find the variable iv (e.g. for density
 * just use ID) of the (i,j) Dof inside a given cell. the pair i,j identifies
 * a unique Dof among the N^2 Dof in 2D.
 * 
 * Time integration is configurable through parameter file. Allowed
 * possiblities are foward_euler, ssprk2, ssprk3 or ssprk54.
 * 
 * Shock capturing with limiters is a delicate subject.
 * It is disabled by default, but can be enable through parameter
 * limiter_enabled.
 *
 * Our first idea was to implement adapt the work by
 * May, Jameson, "A Spectral Difference Method for the Euler
 * and Navier-Stokes Equations on Unstructured Meshes", AIAA 2006-304
 * http://aero-comlab.stanford.edu/Papers/may.aiaa.06-0304.pdf
 * May/Jameson limiting procedure consists in:
 * 1. for each cell, compute reference state as average HydroState 
 *    (stored in Uaverage)
 * 2. for each cell, compute Umin, Umax as min max HydroState in a neighborhood
 * 3. for each cell, evaluate is the high-order cell-border reconstruction 
 *    must be demoted to linear reconstruction. If any of the 
 *    K=number_of_faces x N reconstructed state violates the TVD criterium,
 *    all the faces reconstruction will be demoted to linear reconstruction.
 *
 * We finaly implement the original idea published in Cockburn and Shu,
 * "The Runge-Kutta Discontinuous Galerkin Method for Conservation Laws V: 
 * MultiDimensinal systems", Journal of Computational Physics, 141, 199-224 (1998).
 *
 * 1. compute cell-average of conservative variables, as well as cell-average gradient
 * of the conservative variables.
 * 2. for each space direction, for each cell, compute 3 state-vector:
 *    a. current cell average gradient times dx
 *    b. backward difference of cell-averaged value (current and neighbor cell)
 *    c. forward  difference of cell-averaged value (current and neighbor cell)
 * 3. (optional) project these 3 vectors in the local characteristics space
 *    (eigenspace of the local flux Jacobian matrix); 
 * 4. for each component, perform a TVB-modified minmod limiting of the 3 values 
 *    to detect if dofs must be modified to a 1st order polynomial approximation
 * 5. if limiting detection, was positive actually perform the 1st order modification
 *    in current cell (with optionally a back-projection in the real space from 
 *    the eigenspace).
 *
 * To enabled the limiting procedure to project data into the characteristics space,
 * use parameter:
 * limiter_characteristics_enabled=true
 * in the sdm section of the ini parameter file.
 *
 * If viscous terms computation is enabled, we need 
 * - Ugrax_v, Ugrady_v (and Ugradz_v) allocated;
 *   these arrays are used to store velocity gradients at soluton points.
 * - FUgrad allocated;
 *   this array is used to store both velocity and velocity gradients at flux points.
 *
 * If thermal_diffusivity_terms_enabled, temperature gradients need to be stored.
 *
 */
template<int dim, int N>
class SolverHydroSDM : public ppkMHD::SolverBase
{

public:

  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim==2,DataArray2dHost,DataArray3dHost>::type;

  //! a type to store some coefficients needed to perform Runge-Kutta integration
  using coefs_t = Kokkos::Array<real_t,3>;

  static constexpr int get_dim() {return dim;};
  static constexpr int get_N()   {return N;};
  
  SolverHydroSDM(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroSDM();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroSDM<dim,N>* solver = new SolverHydroSDM<dim,N>(params, configMap);

    return solver;
  }

  /**
   * Return a string that uniquely identifies a SDM Solver. This class
   * returns a string <tt>SolverHydroSDM<dim,N></tt>, with @p dim and @p N
   * replaced by appropriate template values.
   */
  virtual std::string get_name () const;

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     Uaux;  /*!< auxiliary hydrodynamics conservative variables arrays (used in computing fluxes divergence) */

  DataArray     Uaverage; /*! used if limiting is enabled */
  //DataArray     Umin;     /*! used if limiting is enabled */
  //DataArray     Umax;     /*! used if limiting is enabled */

  /*
   * limiter specific arrays
   */
  DataArray     Ugradx; /*! used if limiting is enabled, cell-averaged gradient, x component */
  DataArray     Ugrady; /*! used if limiting is enabled, cell-averaged gradient, y component */
  DataArray     Ugradz; /*! used if limiting is enabled, cell-averaged gradient, z component */

  /*
   * viscous terms specific array
   */
  DataArray     Ugradx_v; /* velocity gradient-x, used and allocated only if viscous terms enabled */
  DataArray     Ugrady_v; /* velocity gradient-y, used and allocated only if viscous terms enabled */
  DataArray     Ugradz_v; /* velocity gradient-z, used and allocated only if viscous terms enabled */

  DataArray     FUgrad; /* velocity and velocity gradient at flux points */
  
  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray     U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes : intermediate array containing fluxes, used in
  //! compute_fluxes_divergence_per_dir
  DataArray Fluxes;
  
  /*
   * Override base class method to initialize IO writer object
   */
  void init_io_writer();

  //! SDM config
  SDM_Geometry<dim,N> sdm_geom;
    
  //! system of equations
  ppkMHD::EulerEquations<dim> euler;

  /*
   * methods
   */

  //! initialize sdm (geometric terms matrix)
  void init_sdm_geometry();
    
  //! compute time step inside an MPI process, at shared memory level.
  double compute_dt_local();

  //! perform 1 time step (time integration).
  void next_iteration_impl();

  //! numerical scheme
  void time_integration(real_t dt);
  
  //! wrapper to tha actual time integation scheme
  void time_integration_impl(DataArray Udata, 
			     DataArray Udata_fdiv, 
			     real_t dt);

  //! all computation that must be done before all others
  void apply_pre_step_computation(DataArray Udata);
  
  //! apply positivity preserving procedure
  void apply_positivity_preserving(DataArray Udata);
  
  //! apply limiting procedure
  void apply_limiting(DataArray Udata);
  
  //! compute invicid hydro flux divergence per direction
  //! this routine is designed to be called from inside compute_fluxes_divergence
  //! \tparam dir identifies direction (IX, IY or IZ)
  template<int dir>
  void compute_invicid_fluxes_divergence_per_dir(DataArray Udata, 
						 DataArray Udata_fdiv, 
						 real_t    dt);

  //! compute viscous hydro flux divergence per direction
  //! this routine is designed to be called from inside compute_fluxes_divergence
  //! \tparam dir identifies direction (IX, IY or IZ)
  template<int dir>
  void compute_viscous_fluxes_divergence_per_dir(DataArray Udata,
						 DataArray Udata_fdiv, 
						 real_t    dt);

  //! compute velocity gradients at solution points and store them
  //! in global arrays Ugradx_v, Ugrady_v, Ugradz_v.
  //! Please note that these arrays must have at least dim components (for each
  //! component of the velocity) + optionally 1 extra variable to store
  //! temperature gradients.
  //! this routine is the first step towards viscous flux terms computations
  //! \param[in] Udata (conservative variables at solution points)
  //! \param[out] Ugrad (velocity gradient in direction dir, at solution points)
  template<int dir>
  void compute_velocity_gradients(DataArray Udata, DataArray Ugrad);
  
  //! compute flux divergence, the main term to perform the actual update
  //! in one of the Runge-Kutta methods.
  //! Udata is used in input (one of Runge-Kutta sub-step data)
  //! Udata_fdiv is used in,out as an accumulator
  //! so that the actual update will be U_{n+1}=U_{n}-dt*Udata_fdiv
  //! this operator is used in every Runge Kutta time intergrator
  void compute_fluxes_divergence(DataArray Udata, 
				 DataArray Udata_fdiv, 
				 real_t    dt);

  //! time integration using forward Euler method
  void time_int_forward_euler(DataArray Udata, 
			      DataArray Udata_fdiv, 
			      real_t dt);

  //! time integration using SSP RK2
  void time_int_ssprk2(DataArray Udata, 
		       DataArray Udata_fdiv, 
		       real_t dt);
  
  //! time integration using SSP RK3
  void time_int_ssprk3(DataArray Udata, 
		       DataArray Udata_fdiv, 
		       real_t dt);

  //! time integration using SSP RK4
  void time_int_ssprk54(DataArray Udata, 
			DataArray Udata_fdiv, 
			real_t dt);

  //! erase a solution data array
  void erase(DataArray data);

  //! avoid override the base class make_boundary method
  template<FaceIdType faceId>
  void make_boundary_sdm(DataArray  Udata,
			 bool       mhd_enabled);

  //! special boundary condition for the wedge test case
  template<FaceIdType faceId>
  void make_boundary_sdm_wedge(DataArray   Udata,
			       WedgeParams wparams);

  //! special boundary condition for the jet test case
  template<FaceIdType faceId>
  void make_boundary_sdm_jet(DataArray   Udata,
			     JetParams   jparams);

  //! main boundaries routine (this is were serial / mpi switch happens)
  void make_boundaries(DataArray Udata);

  //! here we call boundaries condition for serial execution
  void make_boundaries_sdm_serial(DataArray Udata, bool mhd_enabled);

#ifdef USE_MPI
  //! here we call boundaries condition for mpi execution
  void make_boundaries_sdm_mpi(DataArray Udata, bool mhd_enabled);
#endif // USE_MPI
  
  // host routines (initialization)
  void init_implode(DataArray Udata);
  void init_blast(DataArray Udata);
  void init_four_quadrant(DataArray Udata);
  void init_kelvin_helmholtz(DataArray Udata);
  void init_wedge(DataArray Udata);
  void init_jet(DataArray Udata);
  void init_isentropic_vortex(DataArray Udata);
  
  void save_solution_impl();

  //! debug routine that saves a flux data array (for a given direction)
  // template <int dir>
  // void save_flux();
  
  //! time integration
  bool forward_euler_enabled;
  bool ssprk2_enabled;
  bool ssprk3_enabled;
  bool ssprk54_enabled;

  //! when space order is >=3, and time integration is Runge-Kutta, we may
  //! want to rescale dt, to match time and space order
  bool rescale_dt_enabled;

  //! limiter (for shock capturing features)
  bool limiter_enabled;
  bool limiter_characteristics_enabled;
  
  //! positivity preserving (density + pressure)
  bool positivity_enabled;

  //! viscous terms
  bool viscous_terms_enabled;

  //! thermal diffusivity terms : kappa * rho * cp * gradient(T)
  bool thermal_diffusivity_terms_enabled;
  
  int isize, jsize, ksize, nbCells;

}; // class SolverHydroSDM


// =======================================================
// ==== CLASS SolverHydroSDM IMPL =======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int N>
SolverHydroSDM<dim,N>::SolverHydroSDM(HydroParams& params,
				      ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), Uaux(),
  Fluxes(), 
  sdm_geom(),
  forward_euler_enabled(true),
  ssprk2_enabled(false),
  ssprk3_enabled(false),
  ssprk54_enabled(false),
  rescale_dt_enabled(false),
  limiter_enabled(false),
  limiter_characteristics_enabled(false),
  positivity_enabled(false),
  viscous_terms_enabled(false),
  thermal_diffusivity_terms_enabled(false),
  isize(params.isize),
  jsize(params.jsize),
  ksize(params.ksize),
  nbCells(params.isize*params.jsize)
{

  solver_type = SOLVER_SDM;

  if (dim==3)
    nbCells = params.isize*params.jsize*params.ksize;
  
  m_nCells = nbCells;

  int nb_dof_per_cell = dim==2 ? N*N : N*N*N;
  int nb_dof = params.nbvar * nb_dof_per_cell;

  m_nDofsPerCell = nb_dof_per_cell;
  
  // useful for allocating Fluxes, for conservative variables at flux points
  int nb_dof_flux = dim==2 ? (N+1)*N*params.nbvar : (N+1)*N*N*params.nbvar;
  
  long long int total_mem_size = 0;
  
  /*
   * Viscous terms computations.
   *
   */
  viscous_terms_enabled = (params.settings.mu > 0);
  
  /*
   * Thermal diffusivity terms computations.
   *
   */
  thermal_diffusivity_terms_enabled = (params.settings.kappa > 0);

  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (dim==2) {

    U     = DataArray("U", isize, jsize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    Uaux  = DataArray("Uaux",isize, jsize, nb_dof);
    
    Fluxes = DataArray("Fluxes", isize, jsize, nb_dof_flux);

    total_mem_size += isize*jsize*nb_dof      * sizeof(real_t); // U
    total_mem_size += isize*jsize*nb_dof      * sizeof(real_t); // Uaux
    total_mem_size += isize*jsize*nb_dof_flux * sizeof(real_t); // Fluxes
    
  } else if (dim==3) {

    U     = DataArray("U", isize, jsize, ksize, nb_dof);
    Uhost = Kokkos::create_mirror(U);
    Uaux  = DataArray("Uaux",isize, jsize, ksize, nb_dof);
    
    Fluxes = DataArray("Fluxes", isize, jsize, ksize, nb_dof_flux);

    total_mem_size += isize*jsize*ksize*nb_dof      * sizeof(real_t); // U
    total_mem_size += isize*jsize*ksize*nb_dof      * sizeof(real_t); // Uaux
    total_mem_size += isize*jsize*ksize*nb_dof_flux * sizeof(real_t); // Fluxes

  }

  /*
   * Init Spectral Difference Mmethod geometry
   * (solution + flux points locations)
   */
  init_sdm_geometry();
    
  /*
   * Time integration
   */
  forward_euler_enabled = configMap.getBool("sdm", "forward_euler", true);
  ssprk2_enabled        = configMap.getBool("sdm", "ssprk2", false);
  ssprk3_enabled        = configMap.getBool("sdm", "ssprk3", false);
  ssprk54_enabled       = configMap.getBool("sdm", "ssprk54", false);

  // rescale dt to make time order "match" space order ?
  rescale_dt_enabled    = configMap.getBool("sdm", "rescale_dt_enabled", false);
  
  if (ssprk2_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * sizeof(real_t);
    }
    
  } else if (ssprk3_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * 2 * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * 2 * sizeof(real_t);
    }
    
  } else if (ssprk54_enabled) {

    if (dim == 2) {
      U_RK1 = DataArray("U_RK1",isize, jsize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, nb_dof);
      U_RK3 = DataArray("U_RK3",isize, jsize, nb_dof);
      U_RK4 = DataArray("U_RK4",isize, jsize, nb_dof);
      total_mem_size += isize*jsize*nb_dof * 4 * sizeof(real_t);
    } else if (dim == 3) {
      U_RK1 = DataArray("U_RK1",isize, jsize, ksize, nb_dof);
      U_RK2 = DataArray("U_RK2",isize, jsize, ksize, nb_dof);
      U_RK3 = DataArray("U_RK3",isize, jsize, ksize, nb_dof);
      U_RK4 = DataArray("U_RK4",isize, jsize, ksize, nb_dof);
      total_mem_size += isize*jsize*ksize*nb_dof * 4 * sizeof(real_t);
    }
    
  }

  /*
   * viscous terms arrays memory allocation: Ugradx_v, Ugrady_v, Ugradz_v
   */
  if (viscous_terms_enabled) {

    int nb_solutions_pts = dim==2 ? N*N : N*N*N;

    // allocate one variable per dim (vx,vy,vz) per solution point
    int nb_components = dim * nb_solutions_pts;

    // if thermal diffusivity is enabled, we need to store temperature gradient
    // at solution points
    if (thermal_diffusivity_terms_enabled)
      nb_components += nb_solutions_pts;
    
    // memory allocation to store velocity gradients at solution points
    if (dim==2) {
      Ugradx_v   = DataArray("Ugradx_v"    ,isize,jsize,nb_components);
      Ugrady_v   = DataArray("Ugrady_v"    ,isize,jsize,nb_components);
      total_mem_size += isize*jsize*nb_components * 2 * sizeof(real_t);
    } else if (dim==3) {
      Ugradx_v   = DataArray("Ugradx_v"    ,isize,jsize,ksize,nb_components);
      Ugrady_v   = DataArray("Ugrady_v"    ,isize,jsize,ksize,nb_components);
      Ugradz_v   = DataArray("Ugradz_v"    ,isize,jsize,ksize,nb_components);
      total_mem_size += isize*jsize*ksize*nb_components * 3 * sizeof(real_t);
    }

    // number of flux points, as used to address array FUgrad
    int nb_flux_pts = dim==2 ? (N+1)*N : (N+1)*N*N;

    // number of components to address FUgrad : dim for velocity + dim*dim for velocity gradients (tensor)
    int nb_components_FUgrad = dim + dim*dim; // that is 6 in 2D and 12 in 3D

    // memory allocation for FUgrad
    if (dim==2)
      FUgrad = DataArray("FUgrad", isize, jsize,        nb_flux_pts*nb_components_FUgrad);
    else
      FUgrad = DataArray("FUgrad", isize, jsize, ksize, nb_flux_pts*nb_components_FUgrad);
    
  }
  
  /*
   * limiter
   */
  limiter_enabled = configMap.getBool("sdm", "limiter_enabled", false);

  /*
   * Ugradx / Ugrady / Ugradz memory allocation
   */
  if (limiter_enabled) {
    
    // memory allocation to store cell-averaged gradient components
    if (dim==2) {
      Ugradx   = DataArray("Ugradx"    ,isize,jsize,params.nbvar);
      Ugrady   = DataArray("Ugrady"    ,isize,jsize,params.nbvar);
      total_mem_size += isize*jsize*params.nbvar * 2 * sizeof(real_t);
    } else if (dim==3) {
      Ugradx   = DataArray("Ugradx"    ,isize,jsize,ksize,params.nbvar);
      Ugrady   = DataArray("Ugrady"    ,isize,jsize,ksize,params.nbvar);
      Ugradz   = DataArray("Ugradz"    ,isize,jsize,ksize,params.nbvar);
      total_mem_size += isize*jsize*ksize*params.nbvar * 3 * sizeof(real_t);
    }

  }
  
  limiter_characteristics_enabled = configMap.getBool("sdm", "limiter_characteristics_enabled", false);
    
  /*
   * Data arrary Uaverage is used in both positivity preserving and
   * limiter
   */
  positivity_enabled = configMap.getBool("sdm", "positivity_enabled", false);


  if (positivity_enabled or limiter_enabled) {

    if (dim==2) {
      Uaverage = DataArray("Uaverage",isize,jsize,params.nbvar);
      total_mem_size += isize*jsize*params.nbvar * 1 * sizeof(real_t);
    } else if (dim==3) {
      Uaverage = DataArray("Uaverage",isize,jsize,ksize,params.nbvar);
      total_mem_size += isize*jsize*ksize*params.nbvar * 1 * sizeof(real_t);
    }
    
  }
  
  /*
   * initialize hydro array at t=0
   */
  if ( !m_problem_name.compare("implode") ) {

    init_implode(U);

  } else if ( !m_problem_name.compare("blast") ) {

    init_blast(U);

  } else if ( !m_problem_name.compare("four_quadrant") ) {

    init_four_quadrant(U);

  // } else if ( !m_problem_name.compare("kelvin-helmholtz") or
  // 	      !m_problem_name.compare("kelvin_helmholtz")) {

  //   init_kelvin_helmholtz(U);

  } else if ( !m_problem_name.compare("wedge") ) {
    
    init_wedge(U);
    
  } else if ( !m_problem_name.compare("jet") ) {
    
    init_jet(U);
    
  } else if ( !m_problem_name.compare("isentropic_vortex") ) {

    init_isentropic_vortex(U);
    
  } else {

    std::cout << "Problem : " << m_problem_name
	      << " is not recognized / implemented."
	      << std::endl;
    std::cout <<  "Use default - Four Quadrant" << std::endl;
    init_implode(U);
    //init_four_quadrant(U);

  }

  /*
   * Border buffer for MPI
   */
#ifdef USE_MPI
  const int gw = params.ghostWidth;
    
  if (params.dimType == TWO_D) {

    Kokkos::resize(borderBufSend_xmin_2d,    gw, jsize, nb_dof);
    Kokkos::resize(borderBufSend_xmax_2d,    gw, jsize, nb_dof);
    Kokkos::resize(borderBufSend_ymin_2d, isize,    gw, nb_dof);
    Kokkos::resize(borderBufSend_ymax_2d, isize,    gw, nb_dof);

    Kokkos::resize(borderBufRecv_xmin_2d,    gw, jsize, nb_dof);
    Kokkos::resize(borderBufRecv_xmax_2d,    gw, jsize, nb_dof);
    Kokkos::resize(borderBufRecv_ymin_2d, isize,    gw, nb_dof);
    Kokkos::resize(borderBufRecv_ymax_2d, isize,    gw, nb_dof);

    total_mem_size += gw    * jsize * nb_dof * 4 * sizeof(real_t);
    total_mem_size += isize * gw    * nb_dof * 4 * sizeof(real_t);

  } else {

    Kokkos::resize(borderBufSend_xmin_3d,    gw, jsize, ksize, nb_dof);
    Kokkos::resize(borderBufSend_xmax_3d,    gw, jsize, ksize, nb_dof);
    Kokkos::resize(borderBufSend_ymin_3d, isize,    gw, ksize, nb_dof);
    Kokkos::resize(borderBufSend_ymax_3d, isize,    gw, ksize, nb_dof);
    Kokkos::resize(borderBufSend_zmin_3d, isize, jsize,    gw, nb_dof);
    Kokkos::resize(borderBufSend_zmax_3d, isize, jsize,    gw, nb_dof);
    
    Kokkos::resize(borderBufRecv_xmin_3d,    gw, jsize, ksize, nb_dof);
    Kokkos::resize(borderBufRecv_xmax_3d,    gw, jsize, ksize, nb_dof);
    Kokkos::resize(borderBufRecv_ymin_3d, isize,    gw, ksize, nb_dof);
    Kokkos::resize(borderBufRecv_ymax_3d, isize,    gw, ksize, nb_dof);
    Kokkos::resize(borderBufRecv_zmin_3d, isize, jsize,    gw, nb_dof);
    Kokkos::resize(borderBufRecv_zmax_3d, isize, jsize,    gw, nb_dof);

    total_mem_size += gw    * jsize * ksize * nb_dof * 4 * sizeof(real_t);
    total_mem_size += isize * gw    * ksize * nb_dof * 4 * sizeof(real_t);
    total_mem_size += isize * jsize * gw    * nb_dof * 4 * sizeof(real_t);
    
  }
#endif // USE_MPI

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (myRank==0) {
    std::cout << "##########################" << "\n";
    std::cout << "Solver is " << m_solver_name << "\n";
    std::cout << "Problem (init condition) is " << m_problem_name << "\n";
    std::cout << "Spectral Difference order (N) : " << N << "\n";
    std::cout << "Time integration is :\n";
    std::cout << "Forward Euler : " << forward_euler_enabled << "\n";
    std::cout << "SSPRK2        : " << ssprk2_enabled << "\n";
    std::cout << "SSPRK3        : " << ssprk3_enabled << "\n";
    std::cout << "SSPRK54       : " << ssprk54_enabled << "\n";
    std::cout << "##########################" << "\n";
    
    // print parameters on screen
    params.print();
    std::cout << "##########################" << "\n";
    std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
    std::cout << "##########################" << "\n";
  }
  
  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);
  
} // SolverHydroSDM::SolverHydroSDM

// =======================================================
// =======================================================
/**
 *
 */
template<int dim, int N>
SolverHydroSDM<dim,N>::~SolverHydroSDM()
{

} // SolverHydroSDM::~SolverHydroSDM

// =======================================================
// =======================================================
template<int dim, int N>
std::string SolverHydroSDM<dim,N>::get_name() const
{

  std::ostringstream buf;
  buf << "SolverHydroSDM<"
      << dim << "," << N << ">";

  return buf.str();
  
} // SolverHydroSDM<dim,N>::get_name

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_io_writer()
{
  
  // install a new IO_Writer sdm-specific
  m_io_writer = std::make_shared<ppkMHD::io::IO_Writer_SDM<dim,N>>(params,
								   configMap,
								   m_variables_names,
								   sdm_geom);

} // SolverHydroSDM<dim,N>::init_io_writer

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_sdm_geometry()
{

  // TODO
  sdm_geom.init(0);
  sdm_geom.init_lagrange_1d();
  
} // SolverHydroSDM<dim,N>::init_sdm_geometry

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step (local to current MPI process)
 *
 * \note
 * The global time step is computed in compute_dt (from base class SolverBase)
 * which actually calls compute_dt_local.
 *
 */
template<int dim, int N>
double SolverHydroSDM<dim,N>::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;
  DataArray Udata;
  
  Udata = U;
  
  using ComputeDtFunctor =
    typename std::conditional<dim==2,
  			      ComputeDt_Functor_2d<N>,
  			      ComputeDt_Functor_3d<N>>::type;
  
  // call device functor
  ComputeDtFunctor computeDtFunctor(params, sdm_geom, euler, Udata);
  Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);
    
  dt = params.settings.cfl/invDt;

  // rescale dt to match the space order N+1
  if (rescale_dt_enabled and N >= 2 and (ssprk3_enabled or ssprk54_enabled))
    dt = pow(dt, (N+1.0)/3.0);
  
  return dt;

} // SolverHydroSDM::compute_dt_local

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::next_iteration_impl()
{

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI
  if (myRank==0) {
    if (m_iteration % params.nlog == 0) {
      //printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iteration,m_dt, m_t);
      printf("time step=%7d (dt=% 10.8g t=% 10.8f)\n",m_iteration,m_dt, m_t);
    }
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
  
} // SolverHydroSDM::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_integration(real_t dt)
{
  
  time_integration_impl(U , Uaux, dt);
  
} // SolverHydroSDM::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of SDM scheme
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_integration_impl(DataArray Udata, 
						  DataArray Udata_fdiv,
						  real_t dt)
{
  
  // fill ghost cell in Udata
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(Udata);
  timers[TIMER_BOUNDARIES]->stop();
      
  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  if (ssprk2_enabled) {
    
    time_int_ssprk2(Udata, Udata_fdiv, dt);

  } else if (ssprk3_enabled) {
    
    time_int_ssprk3(Udata, Udata_fdiv, dt);
    
  } else if (ssprk54_enabled) {
    
    time_int_ssprk54(Udata, Udata_fdiv, dt);
    
  } else {
    
    time_int_forward_euler(Udata, Udata_fdiv, dt);
    
  }
  
  timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroSDM::time_integration_impl

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::apply_pre_step_computation(DataArray Udata)
{

  // if limiter or positivity preserving are enabled,
  // we first compute Uaverage (cell volume averaged conservative variables) 
  if (limiter_enabled or positivity_enabled) {

    // compute Uaverage
    {
      Average_Conservative_Variables_Functor<dim,N> functor(params,
							    sdm_geom,
							    Udata,
							    Uaverage);
      Kokkos::parallel_for(nbCells, functor);
    }
    
  } // end limiter_enabled or positivity_enabled true

} // SolverHydroSDM<dim,N>::apply_pre_step_computation

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::apply_positivity_preserving(DataArray Udata)
{

  if (positivity_enabled) {
    Apply_positivity_Functor_v2<dim,N> functor(params,
					       sdm_geom,
					       Udata,
					       Uaverage);
    Kokkos::parallel_for(nbCells, functor);
  }
  
} // SolverHydroSDM<dim,N>::apply_positivity_preserving

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::apply_limiting(DataArray Udata)
{
  
  // // if limiter is enabled we need to access cell neighborhood for min/max
  // // cell average values
  // if (limiter_enabled) {
  //   // compute Umin / Umax
  //   {
  //     MinMax_Conservative_Variables_Functor<dim,N> functor(params,
  // 							   sdm_geom,
  // 							   Uaverage,
  // 							   Umin,
  // 							   Umax);
  //     Kokkos::parallel_for(nbCells, functor);
  //   }
  // }

  if (limiter_enabled) {

     // we assume here that Uaverage has been computed in routine apply_pre_step_computation
    // we just need to compute cell-average gradient component.
    {
      Average_Gradient_Functor<dim,N,IX> functor(params,
						 sdm_geom,
						 Udata,
						 Ugradx);
      Kokkos::parallel_for(nbCells, functor);
    }

    {
      Average_Gradient_Functor<dim,N,IY> functor(params,
						 sdm_geom,
						 Udata,
						 Ugrady);
      Kokkos::parallel_for(nbCells, functor);
    }

    if (dim == 3) {
      Average_Gradient_Functor<dim,N,IZ> functor(params,
						 sdm_geom,
						 Udata,
						 Ugradz);
      Kokkos::parallel_for(nbCells, functor);
    }

    // retrieve parameter M_TVB (used in the modified minmod routine)
    real_t M_TVB = configMap.getFloat("sdm","M_TVB",40);
    //const real_t dx = this->params.dx;
    //const real_t Mdx2 = M_TVB * dx * dx;
    const real_t Mdx2 = M_TVB;
    
    {
      
      Apply_limiter_Functor<dim,N> functor(params,
					   sdm_geom,
					   euler,
					   Udata,
					   Uaverage,
					   Ugradx,
					   Ugrady,
					   Ugradz,
					   Mdx2);
      Kokkos::parallel_for(nbCells, functor);
    }

  } // end limiter_enabled
  
} // SolverHydroSDM<dim,N>::apply_limiting

// =======================================================
// =======================================================
template<int dim, int N>
template<int dir>
void SolverHydroSDM<dim,N>::compute_invicid_fluxes_divergence_per_dir(DataArray Udata, 
								      DataArray Udata_fdiv,
								      real_t dt)
{

  if (dim==2 and dir==IZ)
    return;
  
  // 1. interpolate conservative variables from solution points to flux points
  {
    
    Interpolate_At_FluxPoints_Functor<dim,N,dir> functor(params,
							 sdm_geom,
							 Udata,
							 Fluxes);
    Kokkos::parallel_for(nbCells, functor);
    
  }
  
  // 2. inplace computation of fluxes along direction <dir> at flux points
  {
    ComputeFluxAtFluxPoints_Functor<dim,N,dir> functor(params,
						       sdm_geom,
						       euler,
						       Fluxes);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // 3. compute derivative and accumulate in Udata_fdiv
  {
    
    Interpolate_At_SolutionPoints_Functor<dim,N,dir> functor(params,
							     sdm_geom,
							     Fluxes,
							     Udata_fdiv);
    Kokkos::parallel_for(nbCells, functor);
    
  }
  
} // SolverHydroSDM<dim,N>::compute_invicid_fluxes_divergence_per_dir

// =======================================================
// =======================================================
template<int dim, int N>
template<int dir>
void SolverHydroSDM<dim,N>::compute_viscous_fluxes_divergence_per_dir(DataArray Udata,
								      DataArray Udata_fdiv,
								      real_t dt)
{

  if (dim==2 and dir==IZ)
    return;

  // here we assume velocity gradients have already been computed
  // i.e. calls to compute_velocity_gradients have been made, that is Ugradx_v, Ugrady_v, Ugradz_v
  // are populated
  
  //
  // Dir X
  //
  if (dir == IX) {

    // 1. interpolate all velocity components from solution to flux points in the given direction
    //    this will fill component IGU, IGV, IGW of FUgrad
    {
      Interpolate_velocities_Sol2Flux_Functor<dim,N,dir> functor(params, sdm_geom,
								 Udata,  FUgrad);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 2. average velocity at cell borders
    {
      Average_velocity_at_cell_borders_Functor<dim,N,dir> functor(params, sdm_geom, FUgrad);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 3.1. interpolate velocity gradients-x from solution points to flux points
    {
      Interpolate_velocity_gradients_Sol2Flux_Functor<dim,N,dir,IX> functor(params, sdm_geom,
									    Ugradx_v, FUgrad);
      Kokkos::parallel_for(nbCells, functor);
    }
    
    // 3.2. interpolate velocity gradients-y from solution points to flux points
    {
      Interpolate_velocity_gradients_Sol2Flux_Functor<dim,N,dir,IY> functor(params, sdm_geom,
									    Ugrady_v, FUgrad);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 3.3. interpolate velocity gradients-z from solution points to flux points
    if (dim==3) {
      Interpolate_velocity_gradients_Sol2Flux_Functor<dim,N,dir,IZ> functor(params, sdm_geom,
									    Ugradz_v, FUgrad);
      Kokkos::parallel_for(nbCells, functor);
    }

    // 4. average velocity gradients at cell border
    {
      //Average_velocity_gradients_at_cell_borders_Functor<dim,N,dir> functor(params, sdm_geom, FUgrad);
      //Kokkos::parallel_for(nbCells, functor);
    }
    
    // 5. Now at last, one can compute viscous fluxes
    {
      //viscous_flux<dim,N,dir> functor(params, sdm_geom, FUgrad, Udata_fdiv);
      //Kokkos::parallel_for(nbCells, functor);
    }
    
  } // end dir IX

} // 

// =======================================================
// =======================================================
template<int dim, int N>
template<int dir>
void SolverHydroSDM<dim,N>::compute_velocity_gradients(DataArray Udata, DataArray Ugrad)
{

  if (dim==2 and dir==IZ)
    return;
  
  // Please note that Fluxes is used as an intermediate data array, containing data at flux points
  
  //
  // VELOCITY GRADIENTS in direction <dir>
  //
  
  // 1. interpolate velocity from solution points to flux points
  {
    Interpolate_velocities_Sol2Flux_Functor<dim,N,dir> functor(params, sdm_geom,
							       Udata,  Fluxes);
    Kokkos::parallel_for(nbCells, functor);
  }

  // 2. average velocity at cell borders
  {
    Average_velocity_at_cell_borders_Functor<dim,N,dir> functor(params, sdm_geom, Fluxes);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // 3. compute derivative along direction <dir> at solution points
  //    using derivative of Lagrange polynomial
  {
    Interp_grad_velocity_at_SolutionPoints_Functor<dim,N,dir> functor(params, sdm_geom, Fluxes, Ugrad);
    Kokkos::parallel_for(nbCells, functor);
  }
  
} // SolverHydroSDM<dim,N>::compute_velocity_gradients
  
// =======================================================
// =======================================================
// //////////////////////////////////////////////////////////
// Compute fluxes divergence (Udata_fdiv)
// Taken the conservation law dU/dt + div F(U) = 0
// where div F(U) = dF/dx + dG/dy + dH/dz, and F(U) is a short
// notation for invicid fluxes + viscous terms + gravitational terms ...
//
// the actual update will be U_{n+1}=U_{n}-dt*Udata_fdiv
// //////////////////////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::compute_fluxes_divergence(DataArray Udata, 
						      DataArray Udata_fdiv,
						      real_t dt)
{

  // Here is the plan:
  // for each direction
  //  1. interpolate conservative variables from solution points to flux points
  //  2. compute inplace flux at flux points (interior + cell borders)
  //     (TODO: apply any flux limiter ?)
  //  3. compute viscous flux + source terms
  //  4. evaluate flux derivatives at solution points and accumulate in Udata_fdiv

  // erase Udata_fdiv
  erase(Udata_fdiv);

  apply_pre_step_computation(Udata);
  
  apply_limiting(Udata);
  
  apply_positivity_preserving(Udata);
  
  compute_invicid_fluxes_divergence_per_dir<IX>(Udata, Udata_fdiv, dt);
  compute_invicid_fluxes_divergence_per_dir<IY>(Udata, Udata_fdiv, dt);
  compute_invicid_fluxes_divergence_per_dir<IZ>(Udata, Udata_fdiv, dt);

  if (viscous_terms_enabled) {
    compute_velocity_gradients<IX>(Udata, Ugradx_v); // results are stored in Ugradx_v
    compute_velocity_gradients<IY>(Udata, Ugrady_v); // results are stored in Ugrady_v
    compute_velocity_gradients<IZ>(Udata, Ugradz_v); // results are stored in Ugradz_v
    
    compute_viscous_fluxes_divergence_per_dir<IX>(Udata, Udata_fdiv, dt);
    compute_viscous_fluxes_divergence_per_dir<IY>(Udata, Udata_fdiv, dt);
    compute_viscous_fluxes_divergence_per_dir<IZ>(Udata, Udata_fdiv, dt);
  }
  
} // SolverHydroSDM<dim,N>::compute_fluxes_divergence

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Forward Euler time integration
// ///////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_forward_euler(DataArray Udata, 
						   DataArray Udata_fdiv,
						   real_t dt)
{
    
  // evaluate flux divergence
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
  
  // perform actual time update in place in Udata: U_{n+1} = U_{n} - dt * Udata_fdiv
  // translated into Udata = 1.0*Udata + 0.0*Udata - dt * Udata_fdiv 
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
} // SolverHydroSDM::time_int_forward_euler

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK2 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 2th order.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * SSP-RK22 (2 stages, 2nd order).
 *
 * The cfl coefficient is 1, i.e.
 *
 * Dt <= cfl Dt_FE
 * where Dt_FE is the forward Euler Dt
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk2(DataArray Udata, 
					    DataArray Udata_fdiv, 
					    real_t dt)
{

  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
    
  // perform actual time update : U_RK1 = 1.0 * U_{n} + 0.0 * U_{n} - dt * Udata_fdiv
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK1, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ================================================================
  // second step :
  // U_{n+1} = 0.5 * U_n + 0.5 * U_RK1 - 0.5 * dt * div_fluxes(U_RK1)
  // ================================================================
  make_boundaries(U_RK1);
  compute_fluxes_divergence(U_RK1, Udata_fdiv, dt);

  {
    coefs_t coefs= {0.5, 0.5, -0.5};    
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, U_RK1, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
} // SolverHydroSDM::time_int_ssprk2

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK3 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 3th order.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * SSP-RK33 (3 stages, 3nd order).
 *
 * Note: This scheme is also call TVD-RK3
 *
 * The cfl coefficient is 1, i.e.
 *
 * Dt <= cfl Dt_FE
 * where Dt_FE is the forward Euler Dt
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk3(DataArray Udata, 
					    DataArray Udata_fdiv, 
					    real_t dt)
{

  // ===============================================
  // first stage : U_RK1 = U_n - dt * div_fluxes(U_n)
  // ===============================================
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
  
  // perform : U_RK1 = 1.0 * U_{n} + 0.0 * U_{n} - dt * Udata_fdiv 
  {
    coefs_t coefs = {1.0, 0.0, -1.0};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK1, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ==============================================================
  // second stage :
  // U_RK2 = 3/4 * U_n + 1/4 * U_RK1 - 1/4 * dt * div_fluxes(U_RK1)
  // ==============================================================
  make_boundaries(U_RK1);
  compute_fluxes_divergence(U_RK1, Udata_fdiv, dt);
  {
    coefs_t coefs = {0.75, 0.25, -0.25};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK2, Udata, U_RK1, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // ================================================================
  // third stage :
  // U_{n+1} = 1/3 * U_n + 2/3 * U_RK2 - 2/3 * dt * div_fluxes(U_RK2)
  // ================================================================
  make_boundaries(U_RK2);
  compute_fluxes_divergence(U_RK2, Udata_fdiv, dt);
  {
    coefs_t coefs = {1.0/3, 2.0/3, -2.0/3};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, U_RK2, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

} // SolverHydroSDM::time_int_ssprk3

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// SSP RK54 time integration
// ///////////////////////////////////////////
/**
 * Strong Stability Preserving Runge-Kutta integration, 4th order, 5 stages.
 *
 * See http://epubs.siam.org/doi/pdf/10.1137/S0036142901389025
 * A NEW CLASS OF OPTIMAL HIGH-ORDER STRONG-STABILITY-PRESERVING
 * TIME DISCRETIZATION METHODS
 * RAYMOND J. SPITERI AND STEVEN J. RUUTH,
 * SIAM J. Numer. Anal, Vol 40, No 2, pp 469-491
 *
 * This scheme is call SSP-RK54
 *
 * see also article "On High Order Strong Stability Preserving Runge–Kutta and Multi Step Time Discretizations", S. Gottlieb, Journal of Scientific Computing, 25(1-2):105-128 · October 2005:
 * https://www.researchgate.net/publication/220395406_On_High_Order_Strong_Stability_Preserving_Runge-Kutta_and_Multi_Step_Time_Discretizations
 * 
 * Additional note:
 * It has been proved that no 4th order RK, 4 stages SSP-RK scheme
 * exists with positive coefficients (Goettlib and Shu, Total variation
 * diminishing Runge-Kutta schemes, Math. Comp., 67 (1998), pp. 73–85.).
 * This means a SSP-RK44 scheme will have negative coefficients, and it
 * requires a flux operator backward in time stable.
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::time_int_ssprk54(DataArray Udata, 
					     DataArray Udata_fdiv, 
					     real_t dt)
{

  // first  index is RK stage index,
  // for each stage, 3 coefficients required
  // except for the last 5th stage, divided into sub-stages
  const real_t rk54_coef[6][3] =
    {
      {1.0,               0.0,               -0.391752226571890}, /*stage1*/
      {0.444370493651235, 0.555629506348765, -0.368410593050371}, /*stage2*/
      {0.620101851488403, 0.379898148511597, -0.251891774271694}, /*stage3*/
      {0.178079954393132, 0.821920045606868, -0.544974750228521}, /*stage4*/
      {0.517231671970585, 0.096059710526147, -0.063692468666290}, /*stage51*/
      {1.0,               0.386708617503269, -0.226007483236906}  /*stage52*/
    };
  
  // ===============================================
  // stage 1:
  // perform : U_RK1 = rk_54[0][0] * U_{n} +
  //                   rk_54[0][1] * U_{n} +
  //                   rk_54[0][2] * dt * Udata_fdiv 
  // ===============================================
  compute_fluxes_divergence(Udata, Udata_fdiv, dt);
  
  {
    const coefs_t coefs = {rk54_coef[0][0],
			   rk54_coef[0][1],
			   rk54_coef[0][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK1, Udata, Udata, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ===============================================
  // stage 2:
  // perform : U_RK2 = rk_54[1][0] * U_{n} +
  //                   rk_54[1][1] * U_RK1 +
  //                   rk_54[1][2] * dt * Udata_fdiv 
  // ===============================================
  make_boundaries(U_RK1);
  compute_fluxes_divergence(U_RK1, Udata_fdiv, dt);
  
  {
    const coefs_t coefs = {rk54_coef[1][0],
			   rk54_coef[1][1],
			   rk54_coef[1][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK2, Udata, U_RK1, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ===============================================
  // stage 3:
  // perform : U_RK3 = rk_54[2][0] * U_{n} +
  //                   rk_54[2][1] * U_RK2 +
  //                   rk_54[2][2] * dt * Udata_fdiv 
  // ===============================================
  make_boundaries(U_RK2);
  compute_fluxes_divergence(U_RK2, Udata_fdiv, dt);
  
  {
    const coefs_t coefs = {rk54_coef[2][0],
			   rk54_coef[2][1],
			   rk54_coef[2][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK3, Udata, U_RK2, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }
  
  // ===============================================
  // stage 4:
  // perform : U_RK4 = rk_54[3][0] * U_{n} +
  //                   rk_54[3][1] * U_RK3 +
  //                   rk_54[3][2] * dt * Udata_fdiv 
  // ===============================================
  make_boundaries(U_RK3);
  compute_fluxes_divergence(U_RK3, Udata_fdiv, dt);
  
  {
    const coefs_t coefs = {rk54_coef[3][0],
			   rk54_coef[3][1],
			   rk54_coef[3][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, U_RK4, Udata, U_RK3, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  // ===============================================
  // stage 5.1:
  // ===============================================
  {
    const coefs_t coefs = {rk54_coef[4][0],
			   rk54_coef[4][1],
			   rk54_coef[4][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, U_RK2, U_RK3, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  
  // ===============================================
  // stage 5.2:
  // ===============================================
  make_boundaries(U_RK4);
  compute_fluxes_divergence(U_RK4, Udata_fdiv, dt);
  {
    const coefs_t coefs = {rk54_coef[5][0],
			   rk54_coef[5][1],
			   rk54_coef[5][2]};
    SDM_Update_RK_Functor<dim,N> functor(params, sdm_geom, Udata, Udata, U_RK4, Udata_fdiv, coefs, dt);
    Kokkos::parallel_for(nbCells, functor);
  }

  //std::cout << "SSP-RK54 is currently partially implemented\n";
  
} // SolverHydroSDM::time_int_ssprk54

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::erase(DataArray data)
{

  SDM_Erase_Functor<dim,N> functor(params, sdm_geom, data);
  
  Kokkos::parallel_for(nbCells, functor);
  
} // SolverHydroSDM<dim,N>::erase

// =======================================================
// =======================================================
template<int dim, int N>
template<FaceIdType faceId>
void SolverHydroSDM<dim,N>::make_boundary_sdm(DataArray   Udata,
					      bool mhd_enabled)
{

  UNUSED(mhd_enabled);
  
  const int ghostWidth=params.ghostWidth;
  int max_size = std::max(params.isize,params.jsize);
  int nbIter = ghostWidth * max_size;

  if (dim==3) {
    max_size = std::max(max_size,params.ksize);
    nbIter = ghostWidth * max_size * max_size;
  }

  {

    MakeBoundariesFunctor_SDM<dim,N,faceId> functor(params, sdm_geom, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }
    
} // SolverHydroSDM<dim,N>::make_boundary_sdm

// =======================================================
// =======================================================
template<int dim, int N>
template<FaceIdType faceId>
void SolverHydroSDM<dim,N>::make_boundary_sdm_wedge(DataArray   Udata,
						    WedgeParams wparams)
{

  const int ghostWidth=params.ghostWidth;
  int max_size = std::max(params.isize,params.jsize);
  int nbIter = ghostWidth * max_size;

  if (dim==3) {
    max_size = std::max(max_size,params.ksize);
    nbIter = ghostWidth * max_size * max_size;
  }

  {

    MakeBoundariesFunctor_SDM_Wedge<dim,N,faceId> functor(params, sdm_geom, wparams, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }
    
} // SolverHydroSDM<dim,N>::make_boundary_sdm_wedge

// =======================================================
// =======================================================
template<int dim, int N>
template<FaceIdType faceId>
void SolverHydroSDM<dim,N>::make_boundary_sdm_jet(DataArray   Udata,
						  JetParams   jparams)
{

  const int ghostWidth=params.ghostWidth;
  int max_size = std::max(params.isize,params.jsize);
  int nbIter = ghostWidth * max_size;

  if (dim==3) {
    max_size = std::max(max_size,params.ksize);
    nbIter = ghostWidth * max_size * max_size;
  }

  {

    MakeBoundariesFunctor_SDM_Jet<dim,N,faceId> functor(params, sdm_geom, jparams, Udata);
    Kokkos::parallel_for(nbIter, functor);

  }
    
} // SolverHydroSDM<dim,N>::make_boundary_sdm_jet

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundaries(DataArray Udata)
{
  
  bool mhd_enabled = false;

#ifdef USE_MPI
  
  make_boundaries_sdm_mpi(Udata, mhd_enabled);

#else

  make_boundaries_sdm_serial(Udata, mhd_enabled);

#endif // USE_MPI
  
} // SolverHydroSDM::make_boundaries

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundaries_sdm_serial(DataArray Udata,
						       bool mhd_enabled)
{

  /*
   * deal with special cases
   */

  // wedge has a different border condition
  if (dim==2 and !m_problem_name.compare("wedge")) {

    WedgeParams wparams(configMap, m_t);
    
    make_boundary_sdm_wedge<FACE_XMIN>(Udata, wparams);
    make_boundary_sdm_wedge<FACE_XMAX>(Udata, wparams);
    make_boundary_sdm_wedge<FACE_YMIN>(Udata, wparams);
    make_boundary_sdm_wedge<FACE_YMAX>(Udata, wparams);

  } else if (dim==2 and !m_problem_name.compare("jet")) {

    JetParams jparams(configMap);
    
    make_boundary_sdm_jet<FACE_XMIN>(Udata, jparams);
    make_boundary_sdm_jet<FACE_XMAX>(Udata, jparams);
    make_boundary_sdm_jet<FACE_YMIN>(Udata, jparams);
    make_boundary_sdm_jet<FACE_YMAX>(Udata, jparams);

  } else {

    /*
     * General case
     */
    
    make_boundary_sdm<FACE_XMIN>(Udata, mhd_enabled);
    make_boundary_sdm<FACE_XMAX>(Udata, mhd_enabled);
    make_boundary_sdm<FACE_YMIN>(Udata, mhd_enabled);
    make_boundary_sdm<FACE_YMAX>(Udata, mhd_enabled);
    
    if (dim==3) {
      
      make_boundary_sdm<FACE_ZMIN>(Udata, mhd_enabled);
      make_boundary_sdm<FACE_ZMAX>(Udata, mhd_enabled);
      
    }

  }
    
} // SolverHydroSDM<dim,N>::make_boundaries_sdm_serial

#ifdef USE_MPI
// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::make_boundaries_sdm_mpi(DataArray Udata,
						    bool mhd_enabled)
{

  using namespace hydroSimu;
  
  // for each direction:
  // 1. copy boundary to MPI buffer
  // 2. send/recv buffer
  // 3. test if BC is BC_PERIODIC / BC_COPY then ... else ..

  if (dim==2) {

    // ======
    // XDIR
    // ======
    copy_boundaries(Udata,XDIR);
    transfert_boundaries_2d(XDIR);
    
    if (params.neighborsBC[X_MIN] == BC_COPY ||
	params.neighborsBC[X_MIN] == BC_PERIODIC) {
      copy_boundaries_back(Udata, XMIN);
    } else {
      make_boundary_sdm<FACE_XMIN>(Udata, mhd_enabled);
    }
    
    if (params.neighborsBC[X_MAX] == BC_COPY ||
	params.neighborsBC[X_MAX] == BC_PERIODIC) {
      copy_boundaries_back(Udata, XMAX);
    } else {
      make_boundary_sdm<FACE_XMAX>(Udata, mhd_enabled);
    }
    
    params.communicator->synchronize();
    
    // ======
    // YDIR
    // ======
    copy_boundaries(Udata,YDIR);
    transfert_boundaries_2d(YDIR);
    
    if (params.neighborsBC[Y_MIN] == BC_COPY ||
	params.neighborsBC[Y_MIN] == BC_PERIODIC) {
      copy_boundaries_back(Udata, YMIN);
    } else {
      make_boundary_sdm<FACE_YMIN>(Udata, mhd_enabled);
    }
    
    if (params.neighborsBC[Y_MAX] == BC_COPY ||
	params.neighborsBC[Y_MAX] == BC_PERIODIC) {
      copy_boundaries_back(Udata, YMAX);
    } else {
      make_boundary_sdm<FACE_YMAX>(Udata, mhd_enabled);
    }
    
    params.communicator->synchronize();
  
  } else {

    // ======
    // XDIR
    // ======
    copy_boundaries(Udata,XDIR);
    transfert_boundaries_3d(XDIR);
    
    if (params.neighborsBC[X_MIN] == BC_COPY ||
	params.neighborsBC[X_MIN] == BC_PERIODIC) {
      copy_boundaries_back(Udata, XMIN);
    } else {
      make_boundary_sdm<FACE_XMIN>(Udata, mhd_enabled);
    }
    
    if (params.neighborsBC[X_MAX] == BC_COPY ||
	params.neighborsBC[X_MAX] == BC_PERIODIC) {
      copy_boundaries_back(Udata, XMAX);
    } else {
      make_boundary_sdm<FACE_XMAX>(Udata, mhd_enabled);
    }
    
    params.communicator->synchronize();
    
    // ======
    // YDIR
    // ======
    copy_boundaries(Udata,YDIR);
    transfert_boundaries_3d(YDIR);
    
    if (params.neighborsBC[Y_MIN] == BC_COPY ||
	params.neighborsBC[Y_MIN] == BC_PERIODIC) {
      copy_boundaries_back(Udata, YMIN);
    } else {
      make_boundary_sdm<FACE_YMIN>(Udata, mhd_enabled);
    }
    
    if (params.neighborsBC[Y_MAX] == BC_COPY ||
	params.neighborsBC[Y_MAX] == BC_PERIODIC) {
      copy_boundaries_back(Udata, YMAX);
    } else {
      make_boundary_sdm<FACE_YMAX>(Udata, mhd_enabled);
    }
    
    params.communicator->synchronize();
    
    // ======
    // ZDIR
    // ======
    copy_boundaries(Udata,ZDIR);
    transfert_boundaries_3d(ZDIR);
    
    if (params.neighborsBC[Z_MIN] == BC_COPY ||
	params.neighborsBC[Z_MIN] == BC_PERIODIC) {
      copy_boundaries_back(Udata, ZMIN);
    } else {
      make_boundary_sdm<FACE_ZMIN>(Udata, mhd_enabled);
    }
    
    if (params.neighborsBC[Z_MAX] == BC_COPY ||
	params.neighborsBC[Z_MAX] == BC_PERIODIC) {
      copy_boundaries_back(Udata, ZMAX);
    } else {
      make_boundary_sdm<FACE_ZMAX>(Udata, mhd_enabled);
    }
    
    params.communicator->synchronize();
    
  } // end 3d
  
} // SolverHydroSDM<dim,N>::make_boundaries_sdm_mpi
#endif // USE_MPI

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_implode(DataArray Udata)
{

  InitImplodeFunctor<dim,N> functor(params, sdm_geom, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);
  
  InitBlastFunctor<dim,N> functor(params, sdm_geom, blastParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroSDM::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant  riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_four_quadrant(DataArray Udata)
{

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);
    
  HydroState2d U0, U1, U2, U3;
  ppkMHD::getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  ppkMHD::primToCons_2D(U0, params.settings.gamma0);
  ppkMHD::primToCons_2D(U1, params.settings.gamma0);
  ppkMHD::primToCons_2D(U2, params.settings.gamma0);
  ppkMHD::primToCons_2D(U3, params.settings.gamma0);
  
  InitFourQuadrantFunctor<dim,N> functor(params, sdm_geom,
					 Udata,
					 U0, U1, U2, U3,
					 xt, yt);
  Kokkos::parallel_for(nbCells, functor);
    
} // init_four_quadrant

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability test.
 * 
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_kelvin_helmholtz(DataArray Udata)
{

  // KHParams khParams = KHParams(configMap);

  // InitKelvinHelmholtzFunctor<dim,N> functor(params,
  // 						 monomialMap.data,
  // 						 khParams,
  // 						 Udata);
  // Kokkos::parallel_for(nbCells, functor);

} // SolverHydroSDM::init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 * 
 * 
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_wedge(DataArray Udata)
{

  WedgeParams wparams(configMap, 0.0);
  
  InitWedgeFunctor<dim,N> functor(params, sdm_geom, wparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_wedge

// =======================================================
// =======================================================
/**
 * 
 * 
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_jet(DataArray Udata)
{
  
  JetParams jparams(configMap);
  
  InitJetFunctor<dim,N> functor(params, sdm_geom, jparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_jet

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template<int dim, int N>
void SolverHydroSDM<dim,N>::init_isentropic_vortex(DataArray Udata)
{

  IsentropicVortexParams iparams(configMap);

  InitIsentropicVortexFunctor<dim,N> functor(params, sdm_geom, iparams, Udata);
  Kokkos::parallel_for(nbCells, functor);
  
} // init_isentropic_vortex

// =======================================================
// =======================================================
template<int dim, int N>
void SolverHydroSDM<dim,N>::save_solution_impl()
{

  timers[TIMER_IO]->start();

  save_data(U,  Uhost, m_times_saved, m_t);
  
  timers[TIMER_IO]->stop();
    
} // SolverHydroSDM::save_solution_impl()

} // namespace sdm

#endif // SOLVER_HYDRO_SDM_H_
