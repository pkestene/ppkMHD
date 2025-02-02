#ifndef SDM_INIT_FOUR_QUADRANT_FUNCTOR_H_
#define SDM_INIT_FOUR_QUADRANT_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/initRiemannConfig2d.h"

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
template <int dim, int N>
class InitFourQuadrantFunctor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;

  static constexpr auto dofMap = DofMap<dim, N>;

  InitFourQuadrantFunctor(HydroParams          params,
                          SDM_Geometry<dim, N> sdm_geom,
                          DataArray            Udata,
                          HydroState2d         U0,
                          HydroState2d         U1,
                          HydroState2d         U2,
                          HydroState2d         U3,
                          real_t               xt,
                          real_t               yt)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , Udata(Udata)
    , U0(U0)
    , U1(U1)
    , U2(U2)
    , U3(U3)
    , xt(xt)
    , yt(yt){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams          params,
        SDM_Geometry<dim, N> sdm_geom,
        DataArray            Udata,
        HydroState2d         U0,
        HydroState2d         U1,
        HydroState2d         U2,
        HydroState2d         U3,
        real_t               xt,
        real_t               yt)
  {
    int nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    InitFourQuadrantFunctor functor(params, sdm_geom, Udata, U0, U1, U2, U3, xt, yt);
    Kokkos::parallel_for("InitFourQuadrantFunctor", nbCells, functor);
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

    int i, j;
    index2coord(index, i, j, isize, jsize);

    // loop over cell DoF's
    for (int idy = 0; idy < N; ++idy)
    {
      for (int idx = 0; idx < N; ++idx)
      {

        // lower left corner
        real_t x = xmin + dx / 2 + (i + nx * i_mpi - ghostWidth) * dx;
        real_t y = ymin + dy / 2 + (j + ny * j_mpi - ghostWidth) * dy;

        // Dof location in real space
        x += this->sdm_geom.solution_pts_1d(idx) * dx;
        y += this->sdm_geom.solution_pts_1d(idy) * dy;

        if (x < xt)
        {
          if (y < yt)
          {
            // quarter 2
            Udata(i, j, dofMap(idx, idy, 0, ID)) = U2[ID];
            Udata(i, j, dofMap(idx, idy, 0, IE)) = U2[IE];
            Udata(i, j, dofMap(idx, idy, 0, IU)) = U2[IU];
            Udata(i, j, dofMap(idx, idy, 0, IV)) = U2[IV];
          }
          else
          {
            // quarter 1
            Udata(i, j, dofMap(idx, idy, 0, ID)) = U1[ID];
            Udata(i, j, dofMap(idx, idy, 0, IE)) = U1[IE];
            Udata(i, j, dofMap(idx, idy, 0, IU)) = U1[IU];
            Udata(i, j, dofMap(idx, idy, 0, IV)) = U1[IV];
          }
        }
        else
        {
          if (y < yt)
          {
            // quarter 3
            Udata(i, j, dofMap(idx, idy, 0, ID)) = U3[ID];
            Udata(i, j, dofMap(idx, idy, 0, IE)) = U3[IE];
            Udata(i, j, dofMap(idx, idy, 0, IU)) = U3[IU];
            Udata(i, j, dofMap(idx, idy, 0, IV)) = U3[IV];
          }
          else
          {
            // quarter 0
            Udata(i, j, dofMap(idx, idy, 0, ID)) = U0[ID];
            Udata(i, j, dofMap(idx, idy, 0, IE)) = U0[IE];
            Udata(i, j, dofMap(idx, idy, 0, IU)) = U0[IU];
            Udata(i, j, dofMap(idx, idy, 0, IV)) = U0[IV];
          }
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

        } // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d

  DataArray    Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;

}; // InitFourQuadrantFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_INIT_FOUR_QUADRANT_FUNCTOR_H_
