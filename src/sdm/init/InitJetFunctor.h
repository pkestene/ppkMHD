#ifndef SDM_INIT_JET_FUNCTOR_H_
#define SDM_INIT_JET_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#  include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif                        // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/problems/JetParams.h"

namespace ppkMHD
{
namespace sdm
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Jet initial condition functor.
 *
 * reference:
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang,
 * Chi-Wang Shu, Journal of Computational Physics, Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 * http://www.sciencedirect.com/science/article/pii/S0021999110004535
 *
 */
template <int dim, int N>
class InitJetFunctor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;

  static constexpr auto dofMap = DofMap<dim, N>;

  InitJetFunctor(HydroParams          params,
                 SDM_Geometry<dim, N> sdm_geom,
                 JetParams            jparams,
                 DataArray            Udata)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , jparams(jparams)
    , Udata(Udata){};

  ~InitJetFunctor(){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, SDM_Geometry<dim, N> sdm_geom, JetParams jParams, DataArray Udata)
  {
    int nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    InitJetFunctor functor(params, sdm_geom, jParams, Udata);
    Kokkos::parallel_for("InitJetFunctor", nbCells, functor);
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

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // loop over cell DoF's
    for (int idy = 0; idy < N; ++idy)
    {
      for (int idx = 0; idx < N; ++idx)
      {

        Udata(i, j, dofMap(idx, idy, 0, ID)) = jparams.rho2;
        Udata(i, j, dofMap(idx, idy, 0, IE)) = jparams.e_tot2;
        Udata(i, j, dofMap(idx, idy, 0, IU)) = jparams.rho_u2;
        Udata(i, j, dofMap(idx, idy, 0, IV)) = jparams.rho_v2;

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

          Udata(i, j, k, dofMap(idx, idy, idz, ID)) = jparams.rho2;
          Udata(i, j, k, dofMap(idx, idy, idz, IE)) = jparams.e_tot2;
          Udata(i, j, k, dofMap(idx, idy, idz, IU)) = jparams.rho_u2;
          Udata(i, j, k, dofMap(idx, idy, idz, IV)) = jparams.rho_v2;
          Udata(i, j, k, dofMap(idx, idy, idz, IW)) = jparams.rho_w2;

        } // end for idx
      } // end for idy
    } // end for idz

  } // end operator () - 3d

  JetParams jparams;
  DataArray Udata;

}; // class InitJetFunctor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_INIT_JET_FUNCTOR_H_
