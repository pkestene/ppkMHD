#ifndef TEST_SDM_FLUX_FUNCTORS_INIT_H_
#define TEST_SDM_FLUX_FUNCTORS_INIT_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/EulerEquations.h"

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
template <int dim, int N, int compare>
class InitTestFluxFunctor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;

  static constexpr auto dofMap = DofMap<dim, N>;

  InitTestFluxFunctor(HydroParams params, SDM_Geometry<dim, N> sdm_geom, DataArray Udata)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , Udata(Udata){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, SDM_Geometry<dim, N> sdm_geom, DataArray Udata)
  {
    int64_t nbCells =
      (dim == 2) ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    InitTestFluxFunctor functor(params, sdm_geom, Udata);
    Kokkos::parallel_for("IniTestFluxFunctor", nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  rho(real_t x, real_t y, real_t z = 0.0) const
  {
    return 1.0 + 0.1 * (x + y + z);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  u(real_t x, real_t y, real_t z = 0.0) const
  {
    return 0.1 * (-y);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  rho_u(real_t x, real_t y, real_t z = 0.0) const
  {
    return rho(x, y, z) * u(x, y, z);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  v(real_t x, real_t y, real_t z = 0.0) const
  {
    return 0.1 * (x);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  rho_v(real_t x, real_t y, real_t z = 0.0) const
  {
    return rho(x, y, z) * v(x, y, z);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  w(real_t x, real_t y, real_t z = 0.0) const
  {
    return 0.1;
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  rho_w(real_t x, real_t y, real_t z = 0.0) const
  {
    return rho(x, y, z) * w(x, y, z);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  p(real_t x, real_t y, real_t z = 0.0) const
  {
    return 0.14;
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  e(real_t x, real_t y, real_t z = 0.0) const
  {
    real_t ekin = 0.5 * rho(x, y, z) *
                  (u(x, y, z) * u(x, y, z) + v(x, y, z) * v(x, y, z) + w(x, y, z) * w(x, y, z));
    return p(x, y, z) / (this->params.settings.gamma0 - 1.0) + ekin;
  }

  /*
   * 2D version.
   */
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // loop over cell DoF's
    for (int idy = 0; idy < N; ++idy)
    {
      for (int idx = 0; idx < N; ++idx)
      {

        // lower left corner
        real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
        real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;

        x += this->sdm_geom.solution_pts_1d(idx) * dx;
        y += this->sdm_geom.solution_pts_1d(idy) * dy;

        if (compare == 1)
        {
          Udata(i, j, dofMap(idx, idy, 0, ID)) -= rho(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IP)) -= e(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IU)) -= rho_u(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IV)) -= rho_v(x, y);
        }
        else
        {
          Udata(i, j, dofMap(idx, idy, 0, ID)) = rho(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IP)) = e(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IU)) = rho_u(x, y);
          Udata(i, j, dofMap(idx, idy, 0, IV)) = rho_v(x, y);
        }

      } // end for idx
    } // end for idy

  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

#ifdef USE_MPI
    const int i_mpi = this->params.myMpiPos[IX];
    const int j_mpi = this->params.myMpiPos[IY];
    const int k_mpi = this->params.myMpiPos[IZ];
#else
    const int i_mpi = 0;
    const int j_mpi = 0;
    const int k_mpi = 0;
#endif

    const int nx = this->params.nx;
    const int ny = this->params.ny;
    const int nz = this->params.nz;

    const real_t xmin = this->params.xmin;
    const real_t ymin = this->params.ymin;
    const real_t zmin = this->params.zmin;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // loop over cell DoF's
    for (int idz = 0; idz < N; ++idz)
    {
      for (int idy = 0; idy < N; ++idy)
      {
        for (int idx = 0; idx < N; ++idx)
        {

          // lower left corner
          real_t x = xmin + (i + nx * i_mpi - ghostWidth) * dx;
          real_t y = ymin + (j + ny * j_mpi - ghostWidth) * dy;
          real_t z = zmin + (k + nz * k_mpi - ghostWidth) * dz;

          x += this->sdm_geom.solution_pts_1d(idx) * dx;
          y += this->sdm_geom.solution_pts_1d(idy) * dy;
          z += this->sdm_geom.solution_pts_1d(idz) * dz;

          if (compare == 1)
          {
            Udata(i, j, k, dofMap(idx, idy, idz, ID)) -= rho(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IP)) -= e(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IU)) -= rho_u(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IV)) -= rho_v(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IW)) -= rho_w(x, y, z);
          }
          else
          {
            Udata(i, j, k, dofMap(idx, idy, idz, ID)) = rho(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IP)) = e(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IU)) = rho_u(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IV)) = rho_v(x, y, z);
            Udata(i, j, k, dofMap(idx, idy, idz, IW)) = rho_w(x, y, z);
          }

        } // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d

  DataArray Udata;

}; // InitTestFluxFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // TEST_SDM_FLUX_FUNCTORS_INIT_H_
