#ifndef MOOD_UPDATE_FUNCTORS_H_
#define MOOD_UPDATE_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/MoodBaseFunctor.h"

namespace mood
{

// =======================================================================
// =======================================================================
/**
 * Given the fluxes array, perform update of Udata (conservative variable
 * array).
 *
 * \tparam dim dimension (2 or 3).
 */
template <int dim>
class UpdateFunctor
{

public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim == 2, HydroState2d, HydroState3d>::type;

  //! Decide at compile-time which data array to use
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  UpdateFunctor(HydroParams params,
                DataArray   UOld,
                DataArray   UNew,
                DataArray   FluxData_x,
                DataArray   FluxData_y,
                DataArray   FluxData_z)
    : params(params)
    , UOld(UOld)
    , UNew(UNew)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    HydroState tmp;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = UOld(i, j, ID);
      tmp[IP] = UOld(i, j, IP);
      tmp[IU] = UOld(i, j, IU);
      tmp[IV] = UOld(i, j, IV);

      tmp[ID] += FluxData_x(i, j, ID);
      tmp[IP] += FluxData_x(i, j, IP);
      tmp[IU] += FluxData_x(i, j, IU);
      tmp[IV] += FluxData_x(i, j, IV);

      tmp[ID] -= FluxData_x(i + 1, j, ID);
      tmp[IP] -= FluxData_x(i + 1, j, IP);
      tmp[IU] -= FluxData_x(i + 1, j, IU);
      tmp[IV] -= FluxData_x(i + 1, j, IV);

      tmp[ID] += FluxData_y(i, j, ID);
      tmp[IP] += FluxData_y(i, j, IP);
      tmp[IU] += FluxData_y(i, j, IU);
      tmp[IV] += FluxData_y(i, j, IV);

      tmp[ID] -= FluxData_y(i, j + 1, ID);
      tmp[IP] -= FluxData_y(i, j + 1, IP);
      tmp[IU] -= FluxData_y(i, j + 1, IU);
      tmp[IV] -= FluxData_y(i, j + 1, IV);

      UNew(i, j, ID) = tmp[ID];
      UNew(i, j, IP) = tmp[IP];
      UNew(i, j, IU) = tmp[IU];
      UNew(i, j, IV) = tmp[IV];

    } // end if

  } // end operator ()

  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    HydroState tmp;

    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth && j < jsize - ghostWidth &&
        i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = UOld(i, j, k, ID);
      tmp[IP] = UOld(i, j, k, IP);
      tmp[IU] = UOld(i, j, k, IU);
      tmp[IV] = UOld(i, j, k, IV);
      tmp[IW] = UOld(i, j, k, IW);

      tmp[ID] += FluxData_x(i, j, k, ID);
      tmp[IP] += FluxData_x(i, j, k, IP);
      tmp[IU] += FluxData_x(i, j, k, IU);
      tmp[IV] += FluxData_x(i, j, k, IV);
      tmp[IW] += FluxData_x(i, j, k, IW);

      tmp[ID] -= FluxData_x(i + 1, j, k, ID);
      tmp[IP] -= FluxData_x(i + 1, j, k, IP);
      tmp[IU] -= FluxData_x(i + 1, j, k, IU);
      tmp[IV] -= FluxData_x(i + 1, j, k, IV);
      tmp[IW] -= FluxData_x(i + 1, j, k, IW);

      tmp[ID] += FluxData_y(i, j, k, ID);
      tmp[IP] += FluxData_y(i, j, k, IP);
      tmp[IU] += FluxData_y(i, j, k, IU);
      tmp[IV] += FluxData_y(i, j, k, IV);
      tmp[IW] += FluxData_y(i, j, k, IW);

      tmp[ID] -= FluxData_y(i, j + 1, k, ID);
      tmp[IP] -= FluxData_y(i, j + 1, k, IP);
      tmp[IU] -= FluxData_y(i, j + 1, k, IU);
      tmp[IV] -= FluxData_y(i, j + 1, k, IV);
      tmp[IW] -= FluxData_y(i, j + 1, k, IW);

      tmp[ID] += FluxData_z(i, j, k, ID);
      tmp[IP] += FluxData_z(i, j, k, IP);
      tmp[IU] += FluxData_z(i, j, k, IU);
      tmp[IV] += FluxData_z(i, j, k, IV);
      tmp[IW] += FluxData_z(i, j, k, IW);

      tmp[ID] -= FluxData_z(i, j, k + 1, ID);
      tmp[IP] -= FluxData_z(i, j, k + 1, IP);
      tmp[IU] -= FluxData_z(i, j, k + 1, IU);
      tmp[IV] -= FluxData_z(i, j, k + 1, IV);
      tmp[IW] -= FluxData_z(i, j, k + 1, IW);

      UNew(i, j, k, ID) = tmp[ID];
      UNew(i, j, k, IP) = tmp[IP];
      UNew(i, j, k, IU) = tmp[IU];
      UNew(i, j, k, IV) = tmp[IV];
      UNew(i, j, k, IW) = tmp[IW];

    } // end if

  } // end operator ()

  HydroParams params;
  DataArray   UOld;
  DataArray   UNew;
  DataArray   FluxData_x;
  DataArray   FluxData_y;
  DataArray   FluxData_z;

}; // UpdateFunctor

// =======================================================================
// =======================================================================
/**
 * Given the fluxes array, perform update of Udata (conservative variable
 * array).
 *
 * \tparam dim dimension (2 or 3).
 */
template <int dim>
class UpdateFunctor_ssprk2
{

public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim == 2, HydroState2d, HydroState3d>::type;

  //! Decide at compile-time which data array to use
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  UpdateFunctor_ssprk2(HydroParams params,
                       DataArray   UOld,
                       DataArray   URK,
                       DataArray   UNew,
                       DataArray   FluxData_x,
                       DataArray   FluxData_y,
                       DataArray   FluxData_z)
    : params(params)
    , UOld(UOld)
    , URK(URK)
    , UNew(UNew)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    HydroState tmp;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = UOld(i, j, ID) + URK(i, j, ID);
      tmp[IP] = UOld(i, j, IP) + URK(i, j, IP);
      tmp[IU] = UOld(i, j, IU) + URK(i, j, IU);
      tmp[IV] = UOld(i, j, IV) + URK(i, j, IV);

      tmp[ID] += FluxData_x(i, j, ID);
      tmp[IP] += FluxData_x(i, j, IP);
      tmp[IU] += FluxData_x(i, j, IU);
      tmp[IV] += FluxData_x(i, j, IV);

      tmp[ID] -= FluxData_x(i + 1, j, ID);
      tmp[IP] -= FluxData_x(i + 1, j, IP);
      tmp[IU] -= FluxData_x(i + 1, j, IU);
      tmp[IV] -= FluxData_x(i + 1, j, IV);

      tmp[ID] += FluxData_y(i, j, ID);
      tmp[IP] += FluxData_y(i, j, IP);
      tmp[IU] += FluxData_y(i, j, IU);
      tmp[IV] += FluxData_y(i, j, IV);

      tmp[ID] -= FluxData_y(i, j + 1, ID);
      tmp[IP] -= FluxData_y(i, j + 1, IP);
      tmp[IU] -= FluxData_y(i, j + 1, IU);
      tmp[IV] -= FluxData_y(i, j + 1, IV);

      UNew(i, j, ID) = 0.5 * tmp[ID];
      UNew(i, j, IP) = 0.5 * tmp[IP];
      UNew(i, j, IU) = 0.5 * tmp[IU];
      UNew(i, j, IV) = 0.5 * tmp[IV];

    } // end if

  } // end operator ()

  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    HydroState tmp;

    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth && j < jsize - ghostWidth &&
        i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = UOld(i, j, k, ID) + URK(i, j, k, ID);
      tmp[IP] = UOld(i, j, k, IP) + URK(i, j, k, IP);
      tmp[IU] = UOld(i, j, k, IU) + URK(i, j, k, IU);
      tmp[IV] = UOld(i, j, k, IV) + URK(i, j, k, IV);
      tmp[IW] = UOld(i, j, k, IW) + URK(i, j, k, IW);

      tmp[ID] += FluxData_x(i, j, k, ID);
      tmp[IP] += FluxData_x(i, j, k, IP);
      tmp[IU] += FluxData_x(i, j, k, IU);
      tmp[IV] += FluxData_x(i, j, k, IV);
      tmp[IW] += FluxData_x(i, j, k, IW);

      tmp[ID] -= FluxData_x(i + 1, j, k, ID);
      tmp[IP] -= FluxData_x(i + 1, j, k, IP);
      tmp[IU] -= FluxData_x(i + 1, j, k, IU);
      tmp[IV] -= FluxData_x(i + 1, j, k, IV);
      tmp[IW] -= FluxData_x(i + 1, j, k, IW);

      tmp[ID] += FluxData_y(i, j, k, ID);
      tmp[IP] += FluxData_y(i, j, k, IP);
      tmp[IU] += FluxData_y(i, j, k, IU);
      tmp[IV] += FluxData_y(i, j, k, IV);
      tmp[IW] += FluxData_y(i, j, k, IW);

      tmp[ID] -= FluxData_y(i, j + 1, k, ID);
      tmp[IP] -= FluxData_y(i, j + 1, k, IP);
      tmp[IU] -= FluxData_y(i, j + 1, k, IU);
      tmp[IV] -= FluxData_y(i, j + 1, k, IV);
      tmp[IW] -= FluxData_y(i, j + 1, k, IW);

      tmp[ID] += FluxData_z(i, j, k, ID);
      tmp[IP] += FluxData_z(i, j, k, IP);
      tmp[IU] += FluxData_z(i, j, k, IU);
      tmp[IV] += FluxData_z(i, j, k, IV);
      tmp[IW] += FluxData_z(i, j, k, IW);

      tmp[ID] -= FluxData_z(i, j, k + 1, ID);
      tmp[IP] -= FluxData_z(i, j, k + 1, IP);
      tmp[IU] -= FluxData_z(i, j, k + 1, IU);
      tmp[IV] -= FluxData_z(i, j, k + 1, IV);
      tmp[IW] -= FluxData_z(i, j, k + 1, IW);

      UNew(i, j, k, ID) = 0.5 * tmp[ID];
      UNew(i, j, k, IP) = 0.5 * tmp[IP];
      UNew(i, j, k, IU) = 0.5 * tmp[IU];
      UNew(i, j, k, IV) = 0.5 * tmp[IV];
      UNew(i, j, k, IW) = 0.5 * tmp[IW];

    } // end if

  } // end operator ()

  HydroParams params;
  DataArray   UOld;
  DataArray   URK;
  DataArray   UNew;
  DataArray   FluxData_x;
  DataArray   FluxData_y;
  DataArray   FluxData_z;

}; // UpdateFunctor_ssprk2

// =======================================================================
// =======================================================================
/**
 * Given the fluxes array, perform update of Udata (conservative variable
 * array).
 *
 * \tparam dim dimension (2 or 3).
 */
template <int dim>
class UpdateFunctor_weight
{

public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim == 2, HydroState2d, HydroState3d>::type;

  //! Decide at compile-time which data array to use
  using DataArray = typename std::conditional<dim == 2, DataArray2d, DataArray3d>::type;

  UpdateFunctor_weight(HydroParams params,
                       DataArray   UOld,
                       DataArray   URK,
                       DataArray   UNew,
                       DataArray   FluxData_x,
                       DataArray   FluxData_y,
                       DataArray   FluxData_z,
                       real_t      weight_uold,
                       real_t      weight_urk,
                       real_t      weight_flux)
    : params(params)
    , UOld(UOld)
    , URK(URK)
    , UNew(UNew)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z)
    , weight_uold(weight_uold)
    , weight_urk(weight_urk)
    , weight_flux(weight_flux){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ghostWidth = params.ghostWidth;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    HydroState tmp;
    HydroState tmp2;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = weight_uold * UOld(i, j, ID) + weight_urk * URK(i, j, ID);
      tmp[IP] = weight_uold * UOld(i, j, IP) + weight_urk * URK(i, j, IP);
      tmp[IU] = weight_uold * UOld(i, j, IU) + weight_urk * URK(i, j, IU);
      tmp[IV] = weight_uold * UOld(i, j, IV) + weight_urk * URK(i, j, IV);

      tmp2[ID] = FluxData_x(i, j, ID);
      tmp2[IP] = FluxData_x(i, j, IP);
      tmp2[IU] = FluxData_x(i, j, IU);
      tmp2[IV] = FluxData_x(i, j, IV);

      tmp2[ID] -= FluxData_x(i + 1, j, ID);
      tmp2[IP] -= FluxData_x(i + 1, j, IP);
      tmp2[IU] -= FluxData_x(i + 1, j, IU);
      tmp2[IV] -= FluxData_x(i + 1, j, IV);

      tmp2[ID] += FluxData_y(i, j, ID);
      tmp2[IP] += FluxData_y(i, j, IP);
      tmp2[IU] += FluxData_y(i, j, IU);
      tmp2[IV] += FluxData_y(i, j, IV);

      tmp2[ID] -= FluxData_y(i, j + 1, ID);
      tmp2[IP] -= FluxData_y(i, j + 1, IP);
      tmp2[IU] -= FluxData_y(i, j + 1, IU);
      tmp2[IV] -= FluxData_y(i, j + 1, IV);

      UNew(i, j, ID) = tmp[ID] + weight_flux * tmp2[ID];
      UNew(i, j, IP) = tmp[IP] + weight_flux * tmp2[IP];
      UNew(i, j, IU) = tmp[IU] + weight_flux * tmp2[IU];
      UNew(i, j, IV) = tmp[IV] + weight_flux * tmp2[IV];

    } // end if

  } // end operator ()

  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    HydroState tmp;
    HydroState tmp2;

    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth && j < jsize - ghostWidth &&
        i >= ghostWidth && i < isize - ghostWidth)
    {

      tmp[ID] = weight_uold * UOld(i, j, k, ID) + weight_urk * URK(i, j, k, ID);
      tmp[IP] = weight_uold * UOld(i, j, k, IP) + weight_urk * URK(i, j, k, IP);
      tmp[IU] = weight_uold * UOld(i, j, k, IU) + weight_urk * URK(i, j, k, IU);
      tmp[IV] = weight_uold * UOld(i, j, k, IV) + weight_urk * URK(i, j, k, IV);
      tmp[IW] = weight_uold * UOld(i, j, k, IW) + weight_urk * URK(i, j, k, IW);

      tmp2[ID] = FluxData_x(i, j, k, ID);
      tmp2[IP] = FluxData_x(i, j, k, IP);
      tmp2[IU] = FluxData_x(i, j, k, IU);
      tmp2[IV] = FluxData_x(i, j, k, IV);
      tmp2[IW] = FluxData_x(i, j, k, IW);

      tmp2[ID] -= FluxData_x(i + 1, j, k, ID);
      tmp2[IP] -= FluxData_x(i + 1, j, k, IP);
      tmp2[IU] -= FluxData_x(i + 1, j, k, IU);
      tmp2[IV] -= FluxData_x(i + 1, j, k, IV);
      tmp2[IW] -= FluxData_x(i + 1, j, k, IW);

      tmp2[ID] += FluxData_y(i, j, k, ID);
      tmp2[IP] += FluxData_y(i, j, k, IP);
      tmp2[IU] += FluxData_y(i, j, k, IU);
      tmp2[IV] += FluxData_y(i, j, k, IV);
      tmp2[IW] += FluxData_y(i, j, k, IW);

      tmp2[ID] -= FluxData_y(i, j + 1, k, ID);
      tmp2[IP] -= FluxData_y(i, j + 1, k, IP);
      tmp2[IU] -= FluxData_y(i, j + 1, k, IU);
      tmp2[IV] -= FluxData_y(i, j + 1, k, IV);
      tmp2[IW] -= FluxData_y(i, j + 1, k, IW);

      tmp2[ID] += FluxData_z(i, j, k, ID);
      tmp2[IP] += FluxData_z(i, j, k, IP);
      tmp2[IU] += FluxData_z(i, j, k, IU);
      tmp2[IV] += FluxData_z(i, j, k, IV);
      tmp2[IW] += FluxData_z(i, j, k, IW);

      tmp2[ID] -= FluxData_z(i, j, k + 1, ID);
      tmp2[IP] -= FluxData_z(i, j, k + 1, IP);
      tmp2[IU] -= FluxData_z(i, j, k + 1, IU);
      tmp2[IV] -= FluxData_z(i, j, k + 1, IV);
      tmp2[IW] -= FluxData_z(i, j, k + 1, IW);

      UNew(i, j, k, ID) = tmp[ID] + weight_flux * tmp2[ID];
      UNew(i, j, k, IP) = tmp[IP] + weight_flux * tmp2[IP];
      UNew(i, j, k, IU) = tmp[IU] + weight_flux * tmp2[IU];
      UNew(i, j, k, IV) = tmp[IV] + weight_flux * tmp2[IV];
      UNew(i, j, k, IW) = tmp[IW] + weight_flux * tmp2[IW];

    } // end if

  } // end operator ()

  HydroParams params;
  DataArray   UOld;
  DataArray   URK;
  DataArray   UNew;
  DataArray   FluxData_x;
  DataArray   FluxData_y;
  DataArray   FluxData_z;
  real_t      weight_uold;
  real_t      weight_urk;
  real_t      weight_flux;

}; // UpdateFunctor_weight

// =======================================================================
// =======================================================================
/**
 * This functor tries to perform update on density, if density or
 * pressure becomes negative, we flag the cells for recompute.
 *
 * We use MoodBasefunctor as base class to inherit method to compute
 * primitive variables.
 */
template <int dim, int degree>
class ComputeMoodFlagsUpdateFunctor : public MoodBaseFunctor<dim, degree>
{

public:
  using typename MoodBaseFunctor<dim, degree>::DataArray;
  using typename MoodBaseFunctor<dim, degree>::HydroState;
  using MonomMap = typename mood::MonomialMap<dim, degree>::MonomMap;

  ComputeMoodFlagsUpdateFunctor(HydroParams params,
                                MonomMap    monomMap,
                                DataArray   Udata,
                                DataArray   Flags,
                                DataArray   FluxData_x,
                                DataArray   FluxData_y,
                                DataArray   FluxData_z)
    : MoodBaseFunctor<dim, degree>(params, monomMap)
    , Udata(Udata)
    , Flags(Flags)
    , FluxData_x(FluxData_x)
    , FluxData_y(FluxData_y)
    , FluxData_z(FluxData_z){};

  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    int i, j;
    index2coord(index, i, j, isize, jsize);

    // set flags to zero
    Flags(i, j, 0) = 0.0;

    real_t flag_tmp = 0.0;

    if (j >= ghostWidth && j < jsize - ghostWidth && i >= ghostWidth && i < isize - ghostWidth)
    {

      real_t rho = Udata(i, j, ID);
      real_t e = Udata(i, j, IP);
      real_t u = Udata(i, j, IU);
      real_t v = Udata(i, j, IV);

      real_t rho_new = rho + FluxData_x(i, j, ID) - FluxData_x(i + 1, j, ID) +
                       FluxData_y(i, j, ID) - FluxData_y(i, j + 1, ID);

      real_t e_new = e + FluxData_x(i, j, IP) - FluxData_x(i + 1, j, IP) + FluxData_y(i, j, IP) -
                     FluxData_y(i, j + 1, IP);

      real_t u_new = u + FluxData_x(i, j, IU) - FluxData_x(i + 1, j, IU) + FluxData_y(i, j, IU) -
                     FluxData_y(i, j + 1, IU);

      real_t v_new = v + FluxData_x(i, j, IV) - FluxData_x(i + 1, j, IV) + FluxData_y(i, j, IV) -
                     FluxData_y(i, j + 1, IV);

      // conservative variable
      HydroState UNew;
      UNew[ID] = rho;
      UNew[IP] = e;
      UNew[IU] = u;
      UNew[IV] = v;

      real_t c;
      // compute pressure from primitive variables
      HydroState QNew;
      this->computePrimitives(UNew, &c, QNew);

      // test if solution is not physically admissible (negative density or pressure)
      if (rho_new < 0 or QNew[IP] < 0)
        flag_tmp = 1.0;

      Flags(i, j, 0) = flag_tmp;

    } // end if

  } // end operator () - 2d

  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // set flags to zero
    Flags(i, j, k, 0) = 0.0;

    real_t flag_tmp = 0.0;

    if (k >= ghostWidth && k < ksize - ghostWidth && j >= ghostWidth && j < jsize - ghostWidth &&
        i >= ghostWidth && i < isize - ghostWidth)
    {

      real_t rho = Udata(i, j, k, ID);
      real_t e = Udata(i, j, k, IP);
      real_t u = Udata(i, j, k, IU);
      real_t v = Udata(i, j, k, IV);
      real_t w = Udata(i, j, k, IW);

      real_t rho_new = rho + FluxData_x(i, j, k, ID) - FluxData_x(i + 1, j, k, ID) +
                       FluxData_y(i, j, k, ID) - FluxData_y(i, j + 1, k, ID) +
                       FluxData_z(i, j, k, ID) - FluxData_z(i, j, k + 1, ID);

      real_t e_new = e + FluxData_x(i, j, k, IP) - FluxData_x(i + 1, j, k, IP) +
                     FluxData_y(i, j, k, IP) - FluxData_y(i, j + 1, k, IP) +
                     FluxData_z(i, j, k, IP) - FluxData_z(i, j, k + 1, IP);

      real_t u_new = u + FluxData_x(i, j, k, IU) - FluxData_x(i + 1, j, k, IU) +
                     FluxData_y(i, j, k, IU) - FluxData_y(i, j + 1, k, IU) +
                     FluxData_z(i, j, k, IU) - FluxData_z(i, j, k + 1, IU);

      real_t v_new = v + FluxData_x(i, j, k, IV) - FluxData_x(i + 1, j, k, IV) +
                     FluxData_y(i, j, k, IV) - FluxData_y(i, j + 1, k, IV) +
                     FluxData_z(i, j, k, IV) - FluxData_z(i, j, k + 1, IV);

      real_t w_new = w + FluxData_x(i, j, k, IW) - FluxData_x(i + 1, j, k, IW) +
                     FluxData_y(i, j, k, IW) - FluxData_y(i, j + 1, k, IW) +
                     FluxData_z(i, j, k, IW) - FluxData_z(i, j, k + 1, IW);

      // conservative variable
      HydroState UNew;
      UNew[ID] = rho;
      UNew[IP] = e;
      UNew[IU] = u;
      UNew[IV] = v;
      UNew[IW] = w;

      real_t c;
      // compute pressure from primitive variables
      HydroState QNew;
      this->computePrimitives(UNew, &c, QNew);

      // test if solution is not physically admissible, i.e.
      // negative density or pressure
      if (rho_new < 0 or QNew[IP] < 0)
        flag_tmp = 1.0;

      Flags(i, j, k, 0) = flag_tmp;

    } // end if

  } // end operator () - 3d

  DataArray Udata;
  DataArray Flags;
  DataArray FluxData_x;
  DataArray FluxData_y;
  DataArray FluxData_z;

}; // ComputeMoodFlagsUpdateFunctor

} // namespace mood

#endif // MOOD_UPDATE_FUNCTORS_H_
