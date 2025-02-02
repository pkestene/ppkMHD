/**
 * class SolverHydroMood implementation.
 */
#ifndef SOLVER_HYDRO_MOOD_H_
#define SOLVER_HYDRO_MOOD_H_

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
#include "shared/BoundariesFunctorsWedge.h"
#include "shared/problems/initRiemannConfig2d.h"

// mood
#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"
#include "mood/Polynomial.h"
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

// mood functors (where the action takes place)
#include "mood/MoodPolynomialReconstructionFunctors.h"
#include "mood/MoodFluxesFunctors.h"
#include "mood/MoodInitFunctors.h"
#include "mood/MoodDtFunctor.h"
#include "mood/MoodUpdateFunctors.h"

// for test / debug only
#include "mood/MoodTestReconstruction.h"

#include "mood/mood_utils.h"

// for IO
#include <utils/io/IO_ReadWrite.h>

// for specific init / border conditions
#include "shared/problems/BlastParams.h"
#include "shared/problems/KHParams.h"
#include "shared/problems/WedgeParams.h"

namespace mood
{

/**
 * Main hydrodynamics data structure driving MOOD numerical solver.
 *
 * \tparam dim dimension of the domain (2 or 3)
 * \tparam degree od the polynomial used to reconstruct values at
 *  quadrature points.
 */
template <int dim, int degree>
class SolverHydroMood : public ppkMHD::SolverBase
{

public:
  //! Decide at compile-time which data array to use for 2d or 3d
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  //! Data array typedef for host memory space
  using DataArrayHost = typename std::conditional<dim == 2, DataArray2dHost, DataArray3dHost>::type;

  //! total number of coefficients in the polynomial
  static const int ncoefs = mood::binomial<dim + degree, dim>();

  /**
   * stencilId.
   * This is really ugly because nvcc does'nt support 2d array in constexpr
   */
  // static constexpr STENCIL_ID stencilId = STENCIL_MAP[dim-2][degree-1];
  static constexpr STENCIL_ID stencilId = STENCIL_MAPP[(dim - 2) * 5 + degree - 1];

  //! stencil size (number of cells)
  static const int stencil_size = STENCIL_SIZE[stencilId];

  SolverHydroMood(HydroParams & params, ConfigMap & configMap);
  virtual ~SolverHydroMood();

  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase *
  create(HydroParams & params, ConfigMap & configMap)
  {
    SolverHydroMood<dim, degree> * solver = new SolverHydroMood<dim, degree>(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */

  //! reconstructing polynomial
  Kokkos::Array<DataArray, ncoefs> PolyCoefs;

  //! Runge-Kutta temporary array (will be allocated only if necessary)
  DataArray U_RK1, U_RK2, U_RK3, U_RK4;

  //! fluxes
  DataArray Fluxes_x, Fluxes_y, Fluxes_z;

  //! mood detection
  DataArray MoodFlags;

  /*
   * MOOD config
   */
  Stencil stencil;

  //! ordered list of monomials
  MonomialMap<dim, degree> monomialMap;

  Matrix geomMatrix;

  //! pseudo-inverse of the geomMatrix
  mood_matrix_pi_t      geomMatrixPI_view;
  mood_matrix_pi_host_t geomMatrixPI_view_h;

  //! Quadrature point location view
  QuadLoc_2d_t   QUAD_LOC_2D;
  QuadLoc_2d_h_t QUAD_LOC_2D_h;

  QuadLoc_3d_t   QUAD_LOC_3D;
  QuadLoc_3d_h_t QUAD_LOC_3D_h;

  /*
   * methods
   */

  //! initialize mood (geometric terms matrix)
  void
  init_mood();

  //! initialize quadrature rules in 2d
  void
  init_quadrature_2d();

  //! initialize quadrature rules in 2d
  void
  init_quadrature_3d();

  //! compute time step inside an MPI process, at shared memory level.
  double
  compute_dt_local();

  //! perform 1 time step (time integration).
  void
  next_iteration_impl();

  //! numerical scheme
  void
  time_integration(real_t dt);

  //! wrapper to tha actual time integation scheme
  void
  time_integration_impl(DataArray data_in, DataArray data_out, real_t dt);

  //! time integration using forward Euler method
  void
  time_int_forward_euler(DataArray data_in, DataArray data_out, real_t dt);

  //! time integration using SSP RK2
  void
  time_int_ssprk2(DataArray data_in, DataArray data_out, real_t dt);

  //! time integration using SSP RK3
  void
  time_int_ssprk3(DataArray data_in, DataArray data_out, real_t dt);

  //! time integration using SSP RK4
  void
  time_int_ssprk54(DataArray data_in, DataArray data_out, real_t dt);


  template <int dim_ = dim>
  void
  make_boundaries(typename std::enable_if<dim_ == 2, DataArray2d>::type Udata);

  template <int dim_ = dim>
  void
  make_boundaries(typename std::enable_if<dim_ == 3, DataArray3d>::type Udata);

  // host routines (initialization)
  void
  init_implode(DataArray Udata);
  void
  init_blast(DataArray Udata);
  void
  init_four_quadrant(DataArray Udata);
  void
  init_kelvin_helmholtz(DataArray Udata);
  void
  init_wedge(DataArray Udata);
  void
  init_isentropic_vortex(DataArray Udata);

  void
  save_solution_impl();

  // time integration
  bool forward_euler_enabled;
  bool ssprk2_enabled;
  bool ssprk3_enabled;
  bool ssprk54_enabled;

  int isize, jsize, ksize, nbCells;

}; // class SolverHydroMood


// =======================================================
// ==== CLASS SolverHydroMood IMPL =======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
template <int dim, int degree>
SolverHydroMood<dim, degree>::SolverHydroMood(HydroParams & params, ConfigMap & configMap)
  : SolverBase(params, configMap)
  , U()
  , Uhost()
  , U2()
  , Fluxes_x()
  , Fluxes_y()
  , Fluxes_z()
  , MoodFlags()
  , isize(params.isize)
  , jsize(params.jsize)
  , ksize(params.ksize)
  , nbCells(params.isize * params.jsize)
  , stencil(stencilId)
  , monomialMap()
  , geomMatrix(stencil_size - 1, ncoefs - 1)
  , forward_euler_enabled(true)
  , ssprk2_enabled(false)
  , ssprk3_enabled(false)
  , ssprk54_enabled(false)
{

  solver_type = SOLVER_MOOD;

  if (dim == 3)
    nbCells = params.isize * params.jsize * params.ksize;

  m_nCells = nbCells;
  m_nDofsPerCell = 1;

  int nbvar = params.nbvar;

  long long int total_mem_size = 0;

  /*
   * memory allocation (use sizes with ghosts included)
   */
  if (dim == 2)
  {

    U = DataArray("U", isize, jsize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2 = DataArray("U2", isize, jsize, nbvar);

    Fluxes_x = DataArray("Fluxes_x", isize, jsize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, nbvar);
    MoodFlags = DataArray("MoodFlags", isize, jsize, 1);

    // init polynomial coefficients array
    for (int ip = 0; ip < ncoefs; ++ip)
    {
      std::string label = "PolyCoefs_" + std::to_string(ip);
      PolyCoefs[ip] = DataArray(label, isize, jsize, nbvar);
    }

    total_mem_size += isize * jsize * nbvar * 4 * sizeof(real_t);
    total_mem_size += isize * jsize * sizeof(real_t);
    total_mem_size += isize * jsize * nbvar * ncoefs * sizeof(real_t);
  }
  else if (dim == 3)
  {

    U = DataArray("U", isize, jsize, ksize, nbvar);
    Uhost = Kokkos::create_mirror(U);
    U2 = DataArray("U2", isize, jsize, ksize, nbvar);

    Fluxes_x = DataArray("Fluxes_x", isize, jsize, ksize, nbvar);
    Fluxes_y = DataArray("Fluxes_y", isize, jsize, ksize, nbvar);
    Fluxes_z = DataArray("Fluxes_z", isize, jsize, ksize, nbvar);
    MoodFlags = DataArray("MoodFlags", isize, jsize, ksize, 1);

    // init polynomial coefficients array
    for (int ip = 0; ip < ncoefs; ++ip)
    {
      std::string label = "PolyCoefs_" + std::to_string(ip);
      PolyCoefs[ip] = DataArray(label, isize, jsize, ksize, nbvar);
    }

    total_mem_size += isize * jsize * ksize * nbvar * 5 * sizeof(real_t);
    total_mem_size += isize * jsize * ksize * sizeof(real_t);
    total_mem_size += isize * jsize * ksize * nbvar * ncoefs * sizeof(real_t);
  }

  /*
   * Init MOOD structure (geometric terms matrix and its pseudo invers).
   */
  init_mood();

  /*
   * quadrature rules initialization.
   */
  init_quadrature_2d();
  init_quadrature_3d();

  /*
   * Time integration
   */
  forward_euler_enabled = configMap.getBool("mood", "forward_euler", true);
  ssprk2_enabled = configMap.getBool("mood", "ssprk2", false);
  ssprk3_enabled = configMap.getBool("mood", "ssprk3", false);
  ssprk54_enabled = configMap.getBool("mood", "ssprk54", false);

  if (ssprk2_enabled)
  {

    if (dim == 2)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, nbvar);
      total_mem_size += isize * jsize * nbvar * sizeof(real_t);
    }
    else if (dim == 3)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, ksize, nbvar);
      total_mem_size += isize * jsize * ksize * nbvar * sizeof(real_t);
    }
  }
  else if (ssprk3_enabled)
  {

    if (dim == 2)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, nbvar);
      U_RK2 = DataArray("U_RK2", isize, jsize, nbvar);
      total_mem_size += isize * jsize * nbvar * 2 * sizeof(real_t);
    }
    else if (dim == 3)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, ksize, nbvar);
      U_RK2 = DataArray("U_RK2", isize, jsize, ksize, nbvar);
      total_mem_size += isize * jsize * ksize * nbvar * 2 * sizeof(real_t);
    }
  }
  else if (ssprk54_enabled)
  {

    if (dim == 2)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, nbvar);
      U_RK2 = DataArray("U_RK2", isize, jsize, nbvar);
      U_RK3 = DataArray("U_RK3", isize, jsize, nbvar);
      total_mem_size += isize * jsize * nbvar * 3 * sizeof(real_t);
    }
    else if (dim == 3)
    {
      U_RK1 = DataArray("U_RK1", isize, jsize, ksize, nbvar);
      U_RK2 = DataArray("U_RK2", isize, jsize, ksize, nbvar);
      U_RK3 = DataArray("U_RK3", isize, jsize, ksize, nbvar);
      total_mem_size += isize * jsize * ksize * nbvar * 3 * sizeof(real_t);
    }
  }

  /*
   * initialize hydro array at t=0
   */
  if (!m_problem_name.compare("implode"))
  {

    init_implode(U);
  }
  else if (!m_problem_name.compare("blast"))
  {

    init_blast(U);
  }
  else if (!m_problem_name.compare("four_quadrant"))
  {

    init_four_quadrant(U);
  }
  else if (!m_problem_name.compare("kelvin-helmholtz") or
           !m_problem_name.compare("kelvin_helmholtz"))
  {

    init_kelvin_helmholtz(U);
  }
  else if (!m_problem_name.compare("wedge"))
  {

    init_wedge(U);
  }
  else if (!m_problem_name.compare("isentropic_vortex"))
  {

    init_isentropic_vortex(U);
  }
  else
  {

    std::cout << "Problem : " << m_problem_name << " is not recognized / implemented." << std::endl;
    std::cout << "Use default - Four Quadrant" << std::endl;
    init_four_quadrant(U);
  }
  std::cout << "##########################" << "\n";
  std::cout << "Solver is " << m_solver_name << "\n";
  std::cout << "Problem (init condition) is " << m_problem_name << "\n";
  std::cout << "Mood degree : " << degree << "\n";
  std::cout << "Mood polynomial coefficients : " << ncoefs << "\n";
  std::cout << "StencilId is " << StencilUtils::get_stencil_name(stencil.stencilId) << "\n";
  std::cout << "Number of quadrature points : " << QUADRATURE_NUM_POINTS[stencilId] << "\n";
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

  // initialize time step
  compute_dt();

  // initialize boundaries
  make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2, U);

} // SolverHydroMood::SolverHydroMood

// =======================================================
// =======================================================
/**
 *
 */
template <int dim, int degree>
SolverHydroMood<dim, degree>::~SolverHydroMood()
{} // SolverHydroMood::~SolverHydroMood

// =======================================================
// =======================================================
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_mood()
{

  /*
   * Create the geometric terms matrix.
   */
  std::array<real_t, 3> dxyz = { params.dx, params.dy, params.dz };
  fill_geometry_matrix<dim, degree>(geomMatrix, stencil, monomialMap, dxyz);
  geomMatrix.print("geomMatrix");

  /*
   * compute its pseudo inverse
   */
  Matrix geomMatrixPI;
  compute_pseudo_inverse(geomMatrix, geomMatrixPI);
  geomMatrixPI.print("geomMatrix pseudo inverse");

  printf("Compute pseudo inverse of size %d %d\n", geomMatrixPI.m, geomMatrixPI.n);
  geomMatrixPI_view = mood_matrix_pi_t("geomMatrixPI_view", geomMatrixPI.m, geomMatrixPI.n);
  geomMatrixPI_view_h = Kokkos::create_mirror_view(geomMatrixPI_view);

  /*
   * copy geomMatrixPI into a Kokkos::View (geomMatrixPI_view)
   */
  for (int i = 0; i < geomMatrixPI.m; ++i)
  { // loop over stencil point

    for (int j = 0; j < geomMatrixPI.n; ++j)
    { // loop over monomial

      geomMatrixPI_view_h(i, j) = geomMatrixPI(i, j);
    }
  }

  /*
   * upload pseudo-inverse to device memory
   */
  Kokkos::deep_copy(geomMatrixPI_view, geomMatrixPI_view_h);

} // SolverHydroMood<dim,degree>::init_mood

// =======================================================
// =======================================================
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_quadrature_2d()
{

  // memory allocation for quadrature points coordinates on device and host
  QUAD_LOC_2D = QuadLoc_2d_t("QUAD_LOC_2D");

  QUAD_LOC_2D_h = Kokkos::create_mirror_view(QUAD_LOC_2D);

  /*
   * initialize on host
   */

  // 1 quadrature point (items #2 and #3 are not used)

  // along X
  // -X
  QUAD_LOC_2D_h(0, DIR_X, FACE_MIN, 0, IX) = -0.5;
  QUAD_LOC_2D_h(0, DIR_X, FACE_MIN, 0, IY) = 0.0;

  // + X
  QUAD_LOC_2D_h(0, DIR_X, FACE_MAX, 0, IX) = 0.5;
  QUAD_LOC_2D_h(0, DIR_X, FACE_MAX, 0, IY) = 0.0;

  // along Y
  // -Y
  QUAD_LOC_2D_h(0, DIR_Y, FACE_MIN, 0, IX) = 0.0;
  QUAD_LOC_2D_h(0, DIR_Y, FACE_MIN, 0, IY) = -0.5;

  // +Y
  QUAD_LOC_2D_h(0, DIR_Y, FACE_MAX, 0, IX) = 0.0;
  QUAD_LOC_2D_h(0, DIR_Y, FACE_MAX, 0, IY) = 0.5;

  // 2 quadrature points (item #3 is not used)

  // along X
  // -X
  QUAD_LOC_2D_h(1, DIR_X, FACE_MIN, 0, IX) = -0.5;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MIN, 0, IY) = -0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MIN, 1, IX) = -0.5;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MIN, 1, IY) = 0.5 / SQRT_3;

  // +X
  QUAD_LOC_2D_h(1, DIR_X, FACE_MAX, 0, IX) = 0.5;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MAX, 0, IY) = -0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MAX, 1, IX) = 0.5;
  QUAD_LOC_2D_h(1, DIR_X, FACE_MAX, 1, IY) = 0.5 / SQRT_3;

  // along Y
  // -Y
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MIN, 0, IX) = -0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MIN, 0, IY) = -0.5;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MIN, 1, IX) = 0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MIN, 1, IY) = -0.5;

  // +Y
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MAX, 0, IX) = -0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MAX, 0, IY) = 0.5;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MAX, 1, IX) = 0.5 / SQRT_3;
  QUAD_LOC_2D_h(1, DIR_Y, FACE_MAX, 1, IY) = 0.5;

  // 3 quadrature points

  // along X
  // -X
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 0, IX) = -0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 0, IY) = -0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 1, IX) = -0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 1, IY) = 0.0;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 2, IX) = -0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MIN, 2, IY) = 0.5 * SQRT_3_5;

  // +X
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 0, IX) = 0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 0, IY) = -0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 1, IX) = 0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 1, IY) = 0.0;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 2, IX) = 0.5;
  QUAD_LOC_2D_h(2, DIR_X, FACE_MAX, 2, IY) = 0.5 * SQRT_3_5;

  // along Y
  // -Y
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 0, IX) = -0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 0, IY) = -0.5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 1, IX) = 0.0;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 1, IY) = -0.5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 2, IX) = 0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MIN, 2, IY) = -0.5;

  // +Y
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 0, IX) = -0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 0, IY) = 0.5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 1, IX) = 0.0;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 1, IY) = 0.5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 2, IX) = 0.5 * SQRT_3_5;
  QUAD_LOC_2D_h(2, DIR_Y, FACE_MAX, 2, IY) = 0.5;

  // upload in device memory
  Kokkos::deep_copy(QUAD_LOC_2D, QUAD_LOC_2D_h);

} // SolverHydroMood::init_quadrature_2d

// =======================================================
// =======================================================
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_quadrature_3d()
{

  // memory allocation for quadrature points coordinates on device and host
  QUAD_LOC_3D = QuadLoc_3d_t("QUAD_LOC_3D");
  QUAD_LOC_3D_h = Kokkos::create_mirror_view(QUAD_LOC_3D);

  // 1x1=1 quadrature point (items #2 and #3 are not used)

  // along X
  // -X
  QUAD_LOC_3D_h(0, DIR_X, FACE_MIN, 0, IX) = -0.5;
  QUAD_LOC_3D_h(0, DIR_X, FACE_MIN, 0, IY) = 0.0;
  QUAD_LOC_3D_h(0, DIR_X, FACE_MIN, 0, IZ) = 0.0;

  // +X
  QUAD_LOC_3D_h(0, DIR_X, FACE_MAX, 0, IX) = 0.5;
  QUAD_LOC_3D_h(0, DIR_X, FACE_MAX, 0, IY) = 0.0;
  QUAD_LOC_3D_h(0, DIR_X, FACE_MAX, 0, IZ) = 0.0;

  // along Y
  // -Y
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MIN, 0, IX) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MIN, 0, IY) = -0.5;
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MIN, 0, IZ) = 0.0;

  // +Y
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MAX, 0, IX) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MAX, 0, IY) = 0.5;
  QUAD_LOC_3D_h(0, DIR_Y, FACE_MAX, 0, IZ) = 0.0;

  // along Z
  // -Z
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MIN, 0, IX) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MIN, 0, IY) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MIN, 0, IZ) = -0.5;

  // +Z
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MAX, 0, IX) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MAX, 0, IY) = 0.0;
  QUAD_LOC_3D_h(0, DIR_Z, FACE_MAX, 0, IZ) = 0.5;

  // 2x2=4 quadrature points

  for (int j = 0; j < 2; ++j)
  {

    // aj takes values in [-1,1]
    int aj = (2 * j - 1);

    for (int i = 0; i < 2; ++i)
    {

      // ai takes values in [-1,1]
      int ai = (2 * i - 1);

      int index = i + 2 * j;


      // -X
      QUAD_LOC_3D_h(1, DIR_X, FACE_MIN, index, IX) = -0.5;
      QUAD_LOC_3D_h(1, DIR_X, FACE_MIN, index, IY) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_X, FACE_MIN, index, IZ) = 0.5 / SQRT_3 * aj;

      // +X
      QUAD_LOC_3D_h(1, DIR_X, FACE_MAX, index, IX) = 0.5;
      QUAD_LOC_3D_h(1, DIR_X, FACE_MAX, index, IY) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_X, FACE_MAX, index, IZ) = 0.5 / SQRT_3 * aj;

      // -Y
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MIN, index, IX) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MIN, index, IY) = -0.5;
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MIN, index, IZ) = 0.5 / SQRT_3 * aj;

      // +Y
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MAX, index, IX) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MAX, index, IY) = 0.5;
      QUAD_LOC_3D_h(1, DIR_Y, FACE_MAX, index, IZ) = 0.5 / SQRT_3 * aj;

      // -Z
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MIN, index, IX) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MIN, index, IY) = 0.5 / SQRT_3 * aj;
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MIN, index, IZ) = -0.5;

      // +Z
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MAX, index, IX) = 0.5 / SQRT_3 * ai;
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MAX, index, IY) = 0.5 / SQRT_3 * aj;
      QUAD_LOC_3D_h(1, DIR_Z, FACE_MAX, index, IZ) = 0.5;
    }
  }

  // 3x3=9 quadrature points

  for (int j = -1; j < 2; ++j)
  {
    for (int i = -1; i < 2; ++i)
    {

      int index = (i + 1) + 3 * (j + 1);

      // -X
      QUAD_LOC_3D_h(2, DIR_X, FACE_MIN, index, IX) = -0.5;
      QUAD_LOC_3D_h(2, DIR_X, FACE_MIN, index, IY) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_X, FACE_MIN, index, IZ) = 0.5 * SQRT_3_5 * j;

      // +X
      QUAD_LOC_3D_h(2, DIR_X, FACE_MAX, index, IX) = 0.5;
      QUAD_LOC_3D_h(2, DIR_X, FACE_MAX, index, IY) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_X, FACE_MAX, index, IZ) = 0.5 * SQRT_3_5 * j;

      // -Y
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MIN, index, IX) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MIN, index, IY) = -0.5;
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MIN, index, IZ) = 0.5 * SQRT_3_5 * j;

      // +Y
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MAX, index, IX) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MAX, index, IY) = 0.5;
      QUAD_LOC_3D_h(2, DIR_Y, FACE_MAX, index, IZ) = 0.5 * SQRT_3_5 * j;

      // -Z
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MIN, index, IX) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MIN, index, IY) = 0.5 * SQRT_3_5 * j;
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MIN, index, IZ) = -0.5;

      // +Z
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MAX, index, IX) = 0.5 * SQRT_3_5 * i;
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MAX, index, IY) = 0.5 * SQRT_3_5 * j;
      QUAD_LOC_3D_h(2, DIR_Z, FACE_MAX, index, IZ) = 0.5;
    }
  }

  Kokkos::deep_copy(QUAD_LOC_3D, QUAD_LOC_3D_h);

} // SolverHydroMood::init_quadrature_3d

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
template <int dim, int degree>
double
SolverHydroMood<dim, degree>::compute_dt_local()
{

  real_t    dt;
  real_t    invDt = ZERO_F;
  DataArray Udata;

  // which array is the current one ?
  if (m_iteration % 2 == 0)
    Udata = U;
  else
    Udata = U2;

  // typedef computeDtFunctor
  using ComputeDtFunctor = typename std::
    conditional<dim == 2, ComputeDtFunctor2d<degree>, ComputeDtFunctor3d<degree>>::type;

  // call device functor
  ComputeDtFunctor computeDtFunctor(params, monomialMap.data, Udata);
  Kokkos::parallel_reduce(nbCells, computeDtFunctor, invDt);

  dt = params.settings.cfl / invDt;

  // rescale dt to match the space order degree+1
  if (degree >= 2 and ssprk3_enabled)
    dt = pow(dt, (degree + 1.0) / 3.0);

  return dt;

} // SolverHydroMood::compute_dt_local

// =======================================================
// =======================================================
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::next_iteration_impl()
{

  if (m_iteration % 10 == 0)
  {
    // std::cout << "time step=" << m_iteration << " (dt=" << m_dt << ")" << std::endl;
    printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n", m_iteration, m_dt, m_t);
  }

  // output
  if (params.enableOutput)
  {
    if (should_save_solution())
    {

      std::cout << "Output results at time t=" << m_t << " step " << m_iteration << " dt=" << m_dt
                << std::endl;

      save_solution();

    } // end output
  } // end enable output

  // compute new dt
  timers[TIMER_DT]->start();
  compute_dt();
  timers[TIMER_DT]->stop();

  // perform one step integration
  time_integration(m_dt);

} // SolverHydroMood::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_integration(real_t dt)
{

  if (m_iteration % 2 == 0)
  {
    time_integration_impl(U, U2, dt);
  }
  else
  {
    time_integration_impl(U2, U, dt);
  }

} // SolverHydroMood::time_integration

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual CPU computation of MOOD scheme
// ///////////////////////////////////////////
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_integration_impl(DataArray data_in,
                                                    DataArray data_out,
                                                    real_t    dt)
{

  // fill ghost cell in data_in
  timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  timers[TIMER_BOUNDARIES]->stop();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);

  // start main computation
  timers[TIMER_NUM_SCHEME]->start();

  if (ssprk2_enabled)
  {

    time_int_ssprk2(data_in, data_out, dt);
  }
  else if (ssprk3_enabled)
  {

    time_int_ssprk3(data_in, data_out, dt);
  }
  else if (ssprk54_enabled)
  {

    time_int_ssprk54(data_in, data_out, dt);
  }
  else
  {

    time_int_forward_euler(data_in, data_out, dt);
  }

  timers[TIMER_NUM_SCHEME]->stop();

} // SolverHydroMood::time_integration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Forward Euler time integration
// ///////////////////////////////////////////
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_int_forward_euler(DataArray data_in,
                                                     DataArray data_out,
                                                     real_t    dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  // compute reconstruction polynomial coefficients
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, data_in, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);

    // for (int icoef=0; icoef<ncoefs; ++icoef)
    //   save_data_debug(PolyCoefs[icoef], Uhost, m_times_saved-1, m_t,
    //   "poly"+std::to_string(icoef));
  }

  // for debug only
  // {
  //   if (dim==2) {

  //     int nbvar = params.nbvar;

  //     DataArray RecState1 = DataArray("RecState1", isize, jsize, nbvar);
  //     DataArray RecState2 = DataArray("RecState2", isize, jsize, nbvar);
  //     DataArray RecState3 = DataArray("RecState3", isize, jsize, nbvar);

  //     TestReconstructionFunctor<dim,degree,stencilId> functor(data_in, PolyCoefs,
  // 							      RecState1, RecState2, RecState3,
  // 							      params, stencil, geomMatrixPI_view,
  // 							      QUAD_LOC_2D);
  //     Kokkos::parallel_for(nbCells,functor);

  //     save_data_debug(RecState1, Uhost, m_times_saved, m_t, "RecState1");
  //     save_data_debug(RecState2, Uhost, m_times_saved, m_t, "RecState2");

  //   } else {
  //   }

  // }


  // compute fluxes
  {
    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         data_in,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);

    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y");
  }

  // for (int iRecomp=0; iRecomp<5; ++iRecomp) {

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(MoodFlags, Uhost, m_times_saved, m_t, "mood_flags");
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x_after");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y_after");
  }
  //}


  // actual update
  {
    UpdateFunctor<dim> functor(params, data_in, data_out, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

} // SolverHydroMood::time_int_forward_euler

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
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_int_ssprk2(DataArray data_in, DataArray data_out, real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);

  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================
  // compute reconstruction polynomial coefficients of data_in
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, data_in, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);

    // for (int icoef=0; icoef<ncoefs; ++icoef)
    //   save_data_debug(PolyCoefs[icoef], Uhost, m_times_saved-1, m_t,
    //   "poly"+std::to_string(icoef));
  }

  // compute fluxes to update data_in
  {
    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         data_in,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);

    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y");
  }

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(MoodFlags, Uhost, m_times_saved, m_t, "mood_flags");
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x_after");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y_after");
  }

  // update: U_RK1 = data_in + dt*fluxes
  {
    UpdateFunctor<dim> functor(params, data_in, U_RK1, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

  make_boundaries(U_RK1);

  // ==================================================================
  // second step : U_{n+1} = 0.5 * (U_n + U_RK1 + dt * fluxes(U_RK1) )
  // ==================================================================
  // compute reconstruction polynomial coefficients of U_RK1
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, U_RK1, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);
  }

  // compute fluxes to update U_RK1
  {

    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         U_RK1,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK1, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK1, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // actual update
  {
    UpdateFunctor_ssprk2<dim> functor(
      params, data_in, U_RK1, data_out, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

} // SolverHydroMood::time_int_ssprk2

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
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_int_ssprk3(DataArray data_in, DataArray data_out, real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);

  // ==============================================
  // first step : U_RK1 = U_n + dt * fluxes(U_n)
  // ==============================================
  // compute reconstruction polynomial coefficients of data_in
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, data_in, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);

    // for (int icoef=0; icoef<ncoefs; ++icoef)
    //   save_data_debug(PolyCoefs[icoef], Uhost, m_times_saved-1, m_t,
    //   "poly"+std::to_string(icoef));
  }

  // compute fluxes to update data_in
  {
    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         data_in,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);

    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y");
  }

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(MoodFlags, Uhost, m_times_saved, m_t, "mood_flags");
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, data_in, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
    // save_data_debug(Fluxes_x, Uhost, m_times_saved, m_t, "flux_x_after");
    // save_data_debug(Fluxes_y, Uhost, m_times_saved, m_t, "flux_y_after");
  }

  // update: U_RK1 = data_in + dt*fluxes
  {
    UpdateFunctor<dim> functor(params, data_in, U_RK1, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

  make_boundaries(U_RK1);

  // ========================================================================
  // second step : U_RK2 = 3/4 * U_n + 1/4 * U_RK1 + 1/4 * dt * fluxes(U_RK1)
  // ========================================================================
  // compute reconstruction polynomial coefficients of U_RK1
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, U_RK1, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);
  }

  // compute fluxes (U_RK1)
  {

    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         U_RK1,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK1, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK1, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // actual update
  // U_RK2 =  3/4 U_n + 1/4 U_RK1 + 1/4 * dt * Flux(U_RK1)
  {
    UpdateFunctor_weight<dim> functor(
      params, data_in, U_RK1, U_RK2, Fluxes_x, Fluxes_y, Fluxes_z, 0.75, 0.25, 0.25);
    Kokkos::parallel_for(nbCells, functor);
  }

  make_boundaries(U_RK2);

  // ============================================================================
  // thrird step : U_{n+1} = 1/3 * U_n + 2/3 * U_RK2 + 2/3 * dt * fluxes(U_RK2)
  // ============================================================================
  // compute reconstruction polynomial coefficients of U_RK2
  {

    ComputeReconstructionPolynomialFunctor<dim, degree, stencilId> functor(
      params, monomialMap.data, U_RK2, PolyCoefs, stencil, geomMatrixPI_view);
    Kokkos::parallel_for(nbCells, functor);
  }

  // compute fluxes (U_RK2)
  {

    ComputeFluxesFunctor<dim, degree, stencilId> functor(params,
                                                         monomialMap.data,
                                                         U_RK2,
                                                         PolyCoefs,
                                                         Fluxes_x,
                                                         Fluxes_y,
                                                         Fluxes_z,
                                                         stencil,
                                                         geomMatrixPI_view,
                                                         QUAD_LOC_2D,
                                                         QUAD_LOC_3D,
                                                         dtdx,
                                                         dtdy,
                                                         dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // flag cells for which fluxes will need to be recomputed
  // because attemp to update leads to physically invalid values
  // (negative density or pressure)
  {
    ComputeMoodFlagsUpdateFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK2, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z);
    Kokkos::parallel_for(nbCells, functor);
  }

  // recompute fluxes arround flagged cells
  {
    RecomputeFluxesFunctor<dim, degree> functor(
      params, monomialMap.data, U_RK2, MoodFlags, Fluxes_x, Fluxes_y, Fluxes_z, dtdx, dtdy, dtdz);
    Kokkos::parallel_for(nbCells, functor);
  }

  // actual update
  // U_{n+1} =  1/3 U_n + 2/3 U_RK2 + 2/3 * dt * Flux(U_RK2)
  {
    UpdateFunctor_weight<dim> functor(
      params, data_in, U_RK2, data_out, Fluxes_x, Fluxes_y, Fluxes_z, 1.0 / 3, 2.0 / 3, 2.0 / 3);
    Kokkos::parallel_for(nbCells, functor);
  }

} // SolverHydroMood::time_int_ssprk3

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
 * It has been proved that no 4th order RK, 4 stages SSP-RK scheme
 * exists with positive coefficients (Goettlib and Shu, Total variation
 * diminishing Runge-Kutta schemes, Math. Comp., 67 (1998), pp. 73–85.).
 * This means a SSP-RK44 scheme will have negative coefficients, and we need to
 * have a flux operator backward in time stable.
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::time_int_ssprk54(DataArray data_in, DataArray data_out, real_t dt)
{

  real_t dtdx;
  real_t dtdy;
  real_t dtdz;

  dtdx = dt / params.dx;
  dtdy = dt / params.dy;
  dtdz = dt / params.dz;

  Kokkos::deep_copy(U_RK1, data_in);

  std::cout << "SSP-RK54 is currently unimplemented\n";

} // SolverHydroMood::time_int_ssprk54

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
template <int dim, int degree>
template <int dim_>
void
SolverHydroMood<dim, degree>::make_boundaries(
  typename std::enable_if<dim_ == 2, DataArray2d>::type Udata)
{

  const int ghostWidth = params.ghostWidth;
  int       nbIter = ghostWidth * std::max(isize, jsize);

  // wedge has a different border condition
  if (!m_problem_name.compare("wedge"))
  {

    WedgeParams wparams(configMap, m_t);

    // call device functor
    {
      MakeBoundariesFunctor2D_wedge<FACE_XMIN> functor(params, wparams, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    {
      MakeBoundariesFunctor2D_wedge<FACE_XMAX> functor(params, wparams, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }

    {
      MakeBoundariesFunctor2D_wedge<FACE_YMIN> functor(params, wparams, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
    {
      MakeBoundariesFunctor2D_wedge<FACE_YMAX> functor(params, wparams, Udata);
      Kokkos::parallel_for(nbIter, functor);
    }
  }
  else
  {

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
  }

} // SolverHydroMood::make_boundaries

template <int dim, int degree>
template <int dim_>
void
SolverHydroMood<dim, degree>::make_boundaries(
  typename std::enable_if<dim_ == 3, DataArray3d>::type Udata)
{

  const int ghostWidth = params.ghostWidth;

  int max_size = std::max(isize, jsize);
  max_size = std::max(max_size, ksize);
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

} // SolverHydroMood::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_implode(DataArray Udata)
{

  InitImplodeFunctor<dim, degree> functor(params, monomialMap.data, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_blast(DataArray Udata)
{

  BlastParams blastParams = BlastParams(configMap);

  InitBlastFunctor<dim, degree> functor(params, monomialMap.data, blastParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroMood::init_blast

// =======================================================
// =======================================================
/**
 * Four quadrant  riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_four_quadrant(DataArray Udata)
{

  int    configNumber = configMap.getInteger("riemann2d", "config_number", 0);
  real_t xt = configMap.getFloat("riemann2d", "x", 0.8);
  real_t yt = configMap.getFloat("riemann2d", "y", 0.8);

  HydroState2d U0, U1, U2, U3;
  ppkMHD::getRiemannConfig2d(configNumber, U0, U1, U2, U3);

  ppkMHD::primToCons_2D(U0, params.settings.gamma0);
  ppkMHD::primToCons_2D(U1, params.settings.gamma0);
  ppkMHD::primToCons_2D(U2, params.settings.gamma0);
  ppkMHD::primToCons_2D(U3, params.settings.gamma0);

  InitFourQuadrantFunctor<dim, degree> functor(
    params, monomialMap.data, Udata, U0, U1, U2, U3, xt, yt);
  Kokkos::parallel_for(nbCells, functor);

} // init_four_quadrant

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability test.
 *
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_kelvin_helmholtz(DataArray Udata)
{

  KHParams khParams = KHParams(configMap);

  InitKelvinHelmholtzFunctor<dim, degree> functor(params, monomialMap.data, khParams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // SolverHydroMood::init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 *
 *
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_wedge(DataArray Udata)
{

  WedgeParams wparams(configMap, 0.0);

  InitWedgeFunctor<dim, degree> functor(params, monomialMap.data, wparams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // init_wedge

// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::init_isentropic_vortex(DataArray Udata)
{

  IsentropicVortexParams iparams(configMap);

  InitIsentropicVortexFunctor<dim, degree> functor(params, monomialMap.data, iparams, Udata);
  Kokkos::parallel_for(nbCells, functor);

} // init_isentropic_vortex

// =======================================================
// =======================================================
template <int dim, int degree>
void
SolverHydroMood<dim, degree>::save_solution_impl()
{

  timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U, Uhost, m_times_saved, m_t);
  else
    save_data(U2, Uhost, m_times_saved, m_t);

  timers[TIMER_IO]->stop();

} // SolverHydroMood::save_solution_impl()

} // namespace mood

#endif // SOLVER_HYDRO_MOOD_H_
