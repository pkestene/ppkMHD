#ifndef SDM_LIMITER_FUNCTORS_H_
#define SDM_LIMITER_FUNCTORS_H_

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
/**
 * This functor computes the average HydroState in each cell
 * and store the result in Uaverage.
 *
 * The space average is performed using a Gauss-Chebyshev quadrature.
 */
template <int dim, int N>
class Average_Conservative_Variables_Functor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  static constexpr auto dofMap = DofMap<dim, N>;

  Average_Conservative_Variables_Functor(HydroParams          params,
                                         SDM_Geometry<dim, N> sdm_geom,
                                         DataArray            Udata,
                                         DataArray            Uaverage)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , Udata(Udata)
    , Uaverage(Uaverage){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, SDM_Geometry<dim, N> sdm_geom, DataArray Udata, DataArray Uaverage)
  {
    int64_t nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    Average_Conservative_Variables_Functor functor(params, sdm_geom, Udata, Uaverage);
    Kokkos::parallel_for("Average_Conservative_Variables_Functor", nbCells, functor);
  }

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // for each variables
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      real_t tmp = 0.0;

      // perform the Gauss-Chebyshev quadrature

      // for each DoFs
      for (int idy = 0; idy < N; ++idy)
      {
        real_t y = this->sdm_geom.solution_pts_1d(idy);
        real_t wy = sqrt(y - y * y);

        for (int idx = 0; idx < N; ++idx)
        {
          real_t x = this->sdm_geom.solution_pts_1d(idx);
          real_t wx = sqrt(x - x * x);

          tmp += Udata(i, j, dofMap(idx, idy, 0, ivar)) * wx * wy;

        } // for idx
      } // for idy

      // final scaling
      tmp *= (M_PI / N) * (M_PI / N);

      const real_t smallr = this->params.settings.smallr;
      if (ivar == ID)
        Uaverage(i, j, ID) = tmp > smallr ? tmp : smallr;
      else
        Uaverage(i, j, ivar) = tmp;

    } // end for ivar

  } // operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // for each variables
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      real_t tmp = 0.0;

      // perform the Gauss-Chebyshev quadrature

      // for each DoFs
      for (int idz = 0; idz < N; ++idz)
      {
        real_t z = this->sdm_geom.solution_pts_1d(idz);
        real_t wz = sqrt(z - z * z);

        for (int idy = 0; idy < N; ++idy)
        {
          real_t y = this->sdm_geom.solution_pts_1d(idy);
          real_t wy = sqrt(y - y * y);

          for (int idx = 0; idx < N; ++idx)
          {
            real_t x = this->sdm_geom.solution_pts_1d(idx);
            real_t wx = sqrt(x - x * x);

            tmp += Udata(i, j, k, dofMap(idx, idy, idz, ivar)) * wx * wy * wz;

          } // for idx
        } // for idy
      } // for idz

      // final scaling
      tmp *= (M_PI / N) * (M_PI / N) * (M_PI / N);

      const real_t smallr = this->params.settings.smallr;
      if (ivar == ID)
        Uaverage(i, j, k, ID) = tmp > smallr ? tmp : smallr;
      else
        Uaverage(i, j, k, ivar) = tmp;

    } // end for ivar

  } // operator () - 3d

  DataArray Udata;
  DataArray Uaverage;

}; // class Average_Conservative_Variables_Functor

//! this enum is used functor MinMax_Conservative_Variables_Functor
//! to decide if min / max are computed used only face neighbors or should
//! include corners too
enum stencil_minmax_t
{
  STENCIL_MINMAX_NOCORNER = 0,
  STENCIL_MINMAX_CORNER = 1
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes the min and max average HydroState
 * over simple stencil and store the result in Umin, Umax.
 *
 * Remember that Uaverage,Umin,Umax is sized with nbvar per cell.
 */
template <int dim, int N>
class MinMax_Conservative_Variables_Functor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  static constexpr auto dofMap = DofMap<dim, N>;

  /**
   * \param[in]  params contains hydrodynamics parameters
   * \param[in]  sdm_geom contains parameters to init base class functor
   * \param[in]  Uaverage contains
   * \param[out] Umin of Uaverage over stencil
   * \param[out] Umax of Uaverage over stencil
   */
  MinMax_Conservative_Variables_Functor(HydroParams          params,
                                        SDM_Geometry<dim, N> sdm_geom,
                                        DataArray            Uaverage,
                                        DataArray            Umin,
                                        DataArray            Umax,
                                        int                  corner_included = 0)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , Uaverage(Uaverage)
    , Umin(Umin)
    , Umax(Umax)
    , corner_included(corner_included){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams          params,
        SDM_Geometry<dim, N> sdm_geom,
        DataArray            Uaverage,
        DataArray            Umin,
        DataArray            Umax,
        int                  corner_included)
  {
    int64_t nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    MinMax_Conservative_Variables_Functor functor(
      params, sdm_geom, Uaverage, Umin, Umax, corner_included);
    Kokkos::parallel_for("MinMax_Conservative_Variables_Functor", nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  compute_min(real_t val1, real_t val2) const
  {

    return val1 < val2 ? val1 : val2;

  } // compute_min

  KOKKOS_INLINE_FUNCTION
  real_t
  compute_max(real_t val1, real_t val2) const
  {

    return val1 > val2 ? val1 : val2;

  } // compute_max

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    // for each variables
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      // init min / max value with current cell values
      real_t minval = Uaverage(i, j, ivar);
      real_t maxval = minval;
      real_t tmp;
      // read stencil values

      tmp = i > 0 ? Uaverage(i - 1, j, ivar) : Uaverage(i, j, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = i < isize - 1 ? Uaverage(i + 1, j, ivar) : Uaverage(i, j, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = j > 0 ? Uaverage(i, j - 1, ivar) : Uaverage(i, j, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = j < jsize - 1 ? Uaverage(i, j + 1, ivar) : Uaverage(i, j, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      // write back the results
      Umin(i, j, ivar) = minval;
      Umax(i, j, ivar) = maxval;

    } // end for ivar

  } // operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    // for each variables
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      // init min / max value with current cell values
      real_t minval = Uaverage(i, j, k, ivar);
      real_t maxval = minval;
      real_t tmp;
      // read stencil values

      tmp = i > 0 ? Uaverage(i - 1, j, k, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = i < isize - 1 ? Uaverage(i + 1, j, k, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = j > 0 ? Uaverage(i, j - 1, k, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = j < jsize - 1 ? Uaverage(i, j + 1, k, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = k > 0 ? Uaverage(i, j, k - 1, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      tmp = k < ksize - 1 ? Uaverage(i, j, k + 1, ivar) : Uaverage(i, j, k, ivar);
      minval = compute_min(minval, tmp);
      maxval = compute_max(maxval, tmp);

      // write back the results
      Umin(i, j, k, ivar) = minval;
      Umax(i, j, k, ivar) = maxval;

    } // end for ivar

  } // operator () - 3d

  DataArray Uaverage;
  DataArray Umin;
  DataArray Umax;
  int       corner_included;

}; // class MinMax_Conservative_Variables_Functor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes one component of the average HydroState gradient
 * in each cell and store the result in Uaverage.
 *
 * The space average is performed using a Gauss-Chebyshev quadrature.
 *
 * \tparam dir integer to specify direction / component of the gradient
 * dir can IX,IY in 2D or IX,IY,IZ in 3D.
 */
template <int dim, int N, int dir>
class Average_Gradient_Functor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  using typename SDMBaseFunctor<dim, N>::solution_values_t;

  static constexpr auto dofMap = DofMap<dim, N>;

  Average_Gradient_Functor(HydroParams          params,
                           SDM_Geometry<dim, N> sdm_geom,
                           DataArray            Udata,
                           DataArray            Uaverage)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , Udata(Udata)
    , Uaverage(Uaverage){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams params, SDM_Geometry<dim, N> sdm_geom, DataArray Udata, DataArray Uaverage)
  {
    int64_t nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    Average_Gradient_Functor functor(params, sdm_geom, Udata, Uaverage);
    Kokkos::parallel_for("Average_Gradient_Functor", nbCells, functor);
  }

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    if (dir == IX)
    {

      // for each variables
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // a vector of values at solution points
        solution_values_t sol;

        // variable used to accumulate gradient before averaging
        real_t tmp_average = 0.0;

        // load Udata values line by line for current cell
        for (int idy = 0; idy < N; ++idy)
        {

          // compute quadrature  weight for the current line of dofs
          real_t y = this->sdm_geom.solution_pts_1d(idy);
          real_t wy = sqrt(y - y * y);

          // read a line
          for (int idx = 0; idx < N; ++idx)
          {

            sol[idx] = Udata(i, j, dofMap(idx, idy, 0, ivar));
          }

          // compute gradient component at each dof of this line
          for (int idx = 0; idx < N; ++idx)
          {

            // compute gradient using Lagrange polynomial representation
            // remember that sol2sol_derivative_h(idof,idx) is the derivative of
            // the idof-th Lagrange polynomial evaluated at the idx-th solution points
            real_t grad_val = 0;
            for (int idof = 0; idof < N; ++idof)
            {
              grad_val += sol[idof] * this->sdm_geom.sol2sol_derivative(idof, idx);
            }

            // we can now accumulate this grad_val into the average gradient
            real_t x = this->sdm_geom.solution_pts_1d(idx);
            real_t wx = sqrt(x - x * x);

            tmp_average += grad_val * wx * wy;

          } // end for idx

        } // end for idy

        // we swept all the dof, all we need is the final scaling
        tmp_average *= (M_PI / N) * (M_PI / N);

        // store the result
        Uaverage(i, j, ivar) = tmp_average;

      } // end for ivar

    } // end dir == IX

    if (dir == IY)
    {

      // for each variables
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // a vector of values at solution points
        solution_values_t sol;

        // variable used to accumulate gradient before averaging
        real_t tmp_average = 0.0;

        // load Udata values line by line for current cell
        for (int idx = 0; idx < N; ++idx)
        {

          // compute quadrature  weight for the current line of dofs
          real_t x = this->sdm_geom.solution_pts_1d(idx);
          real_t wx = sqrt(x - x * x);

          // read a line
          for (int idy = 0; idy < N; ++idy)
          {

            sol[idy] = Udata(i, j, dofMap(idx, idy, 0, ivar));
          }

          // compute gradient component at each dof of this line
          for (int idy = 0; idy < N; ++idy)
          {

            // compute gradient using Lagrange polynomial representation
            // remember that sol2sol_derivative_h(idof,idy) is the derivative of
            // the idof-th Lagrange polynomial evaluated at the idy-th solution points
            real_t grad_val = 0;
            for (int idof = 0; idof < N; ++idof)
            {
              grad_val += sol[idof] * this->sdm_geom.sol2sol_derivative(idof, idy);
            }

            // we can now accumulate this grad_val into the average gradient
            real_t y = this->sdm_geom.solution_pts_1d(idy);
            real_t wy = sqrt(y - y * y);

            tmp_average += grad_val * wx * wy;

          } // end for idx

        } // end for idy

        // we swept all the dof, all we need is the final scaling
        tmp_average *= (M_PI / N) * (M_PI / N);

        // store the result
        Uaverage(i, j, ivar) = tmp_average;

      } // end for ivar

    } // end dir == IY

  } // operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    if (dir == IX)
    {

      // for each variables
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // a line-vector of values at solution points
        solution_values_t sol;

        // variable used to accumulate gradient before averaging
        real_t tmp_average = 0.0;

        // load Udata values line by line for current cell
        for (int idz = 0; idz < N; ++idz)
        {

          // compute quadrature weight for the current line of dofs
          real_t z = this->sdm_geom.solution_pts_1d(idz);
          real_t wz = sqrt(z - z * z);

          for (int idy = 0; idy < N; ++idy)
          {

            // compute quadrature weight for the current line of dofs
            real_t y = this->sdm_geom.solution_pts_1d(idy);
            real_t wy = sqrt(y - y * y);

            // read a line-vector
            for (int idx = 0; idx < N; ++idx)
              sol[idx] = Udata(i, j, k, dofMap(idx, idy, idz, ivar));

            // compute gradient component at each dof of this line
            for (int idx = 0; idx < N; ++idx)
            {

              // compute gradient using Lagrange polynomial representation
              // remember that sol2sol_derivative_h(idof,idx) is the derivative of
              // the idof-th Lagrange polynomial evaluated at the idx-th solution points
              real_t grad_val = 0;
              for (int idof = 0; idof < N; ++idof)
              {
                grad_val += sol[idof] * this->sdm_geom.sol2sol_derivative(idof, idx);
              }

              // we can now accumulate this grad_val into the average gradient
              real_t x = this->sdm_geom.solution_pts_1d(idx);
              real_t wx = sqrt(x - x * x);

              tmp_average += grad_val * wx * wy * wz;

            } // end for idx

          } // end for idy
        } // end for idz

        // we swept all the dof, all we need is the final scaling
        tmp_average *= (M_PI / N) * (M_PI / N) * (M_PI / N);

        // store the result
        Uaverage(i, j, k, ivar) = tmp_average;

      } // end for ivar

    } // end dir == IX

    if (dir == IY)
    {

      // for each variables
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // a line-vector of values at solution points
        solution_values_t sol;

        // variable used to accumulate gradient before averaging
        real_t tmp_average = 0.0;

        // load Udata values line by line for current cell
        for (int idz = 0; idz < N; ++idz)
        {

          // compute quadrature weight for the current line of dofs
          real_t z = this->sdm_geom.solution_pts_1d(idz);
          real_t wz = sqrt(z - z * z);

          for (int idx = 0; idx < N; ++idx)
          {

            // compute quadrature weight for the current line of dofs
            real_t x = this->sdm_geom.solution_pts_1d(idx);
            real_t wx = sqrt(x - x * x);

            // read a line-vector
            for (int idy = 0; idy < N; ++idy)
              sol[idy] = Udata(i, j, k, dofMap(idx, idy, idz, ivar));

            // compute gradient component at each dof of this line
            for (int idy = 0; idy < N; ++idy)
            {

              // compute gradient using Lagrange polynomial representation
              // remember that sol2sol_derivative_h(idof,idy) is the derivative of
              // the idof-th Lagrange polynomial evaluated at the idx-th solution points
              real_t grad_val = 0;
              for (int idof = 0; idof < N; ++idof)
              {
                grad_val += sol[idof] * this->sdm_geom.sol2sol_derivative(idof, idy);
              }

              // we can now accumulate this grad_val into the average gradient
              real_t y = this->sdm_geom.solution_pts_1d(idy);
              real_t wy = sqrt(y - y * y);

              tmp_average += grad_val * wx * wy * wz;

            } // end for idy

          } // end for idx

        } // end for idz

        // we swept all the dof, all we need is the final scaling
        tmp_average *= (M_PI / N) * (M_PI / N) * (M_PI / N);

        // store the result
        Uaverage(i, j, k, ivar) = tmp_average;

      } // end for ivar

    } // end dir == IY

    if (dir == IZ)
    {

      // for each variables
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // a line-vector of values at solution points
        solution_values_t sol;

        // variable used to accumulate gradient before averaging
        real_t tmp_average = 0.0;

        // load Udata values line by line for current cell
        for (int idx = 0; idx < N; ++idx)
        {

          // compute quadrature weight for the current line of dofs
          real_t x = this->sdm_geom.solution_pts_1d(idx);
          real_t wx = sqrt(x - x * x);

          for (int idy = 0; idy < N; ++idy)
          {

            // compute quadrature weight for the current line of dofs
            real_t y = this->sdm_geom.solution_pts_1d(idy);
            real_t wy = sqrt(y - y * y);

            // read a line-vector
            for (int idz = 0; idz < N; ++idz)
              sol[idz] = Udata(i, j, k, dofMap(idx, idy, idz, ivar));

            // compute gradient component at each dof of this line
            for (int idz = 0; idz < N; ++idz)
            {

              // compute gradient using Lagrange polynomial representation
              // remember that sol2sol_derivative_h(idof,idx) is the derivative of
              // the idof-th Lagrange polynomial evaluated at the idx-th solution points
              real_t grad_val = 0;
              for (int idof = 0; idof < N; ++idof)
              {
                grad_val += sol[idof] * this->sdm_geom.sol2sol_derivative(idof, idz);
              }

              // we can now accumulate this grad_val into the average gradient
              real_t z = this->sdm_geom.solution_pts_1d(idz);
              real_t wz = sqrt(z - z * z);

              tmp_average += grad_val * wx * wy * wz;

            } // end for idz

          } // end for idy

        } // end for idz

        // we swept all the dof, all we need is the final scaling
        tmp_average *= (M_PI / N) * (M_PI / N) * (M_PI / N);

        // store the result
        Uaverage(i, j, k, ivar) = tmp_average;

      } // end for ivar

    } // end dir == IZ

  } // operator () - 3d

  DataArray Udata;
  DataArray Uaverage;

}; // class Average_Gradient_Functor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor applies the limiting procedure to all cells.
 *
 * The limiting procedure is described in Cockburn and Shu,
 * "The Runge-Kutta Discontinuous Galerkin Method for Conservation Laws V:
 * MultiDimensinal systems", Journal of Computational Physics, 141, 199-224
 * (1998).
 *
 * We use as a reference implementation (for Discontinuous Galerkin schemes)
 * the code dflo by Praveen Chandrashekar:
 * https://github.com/cpraveen/dflo
 */
template <int dim, int N>
class Apply_limiter_Functor : public SDMBaseFunctor<dim, N>
{

public:
  using typename SDMBaseFunctor<dim, N>::DataArray;
  using typename SDMBaseFunctor<dim, N>::HydroState;

  // using typename SDMBaseFunctor<dim,N>::solution_values_t;

  static constexpr auto dofMap = DofMap<dim, N>;

  Apply_limiter_Functor(HydroParams                 params,
                        SDM_Geometry<dim, N>        sdm_geom,
                        ppkMHD::EulerEquations<dim> euler,
                        DataArray                   Udata,
                        DataArray                   Uaverage,
                        DataArray                   Ugradx,
                        DataArray                   Ugrady,
                        DataArray                   Ugradz,
                        const real_t                Mdx2)
    : SDMBaseFunctor<dim, N>(params, sdm_geom)
    , euler(euler)
    , Udata(Udata)
    , Uaverage(Uaverage)
    , Ugradx(Ugradx)
    , Ugrady(Ugrady)
    , Ugradz(Ugradz)
    , Mdx2(Mdx2){};

  // static method which does it all: create and execute functor
  static void
  apply(HydroParams                 params,
        SDM_Geometry<dim, N>        sdm_geom,
        ppkMHD::EulerEquations<dim> euler,
        DataArray                   Udata,
        DataArray                   Uaverage,
        DataArray                   Ugradx,
        DataArray                   Ugrady,
        DataArray                   Ugradz,
        const real_t                Mdx2)
  {
    int64_t nbCells =
      dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

    Apply_limiter_Functor functor(
      params, sdm_geom, euler, Udata, Uaverage, Ugradx, Ugrady, Ugradz, Mdx2);
    Kokkos::parallel_for("Apply_limiter_Functor", nbCells, functor);
  }

  /**
   * TVB version of minmod limiter. If Mdx2=0 then it is TVD limiter.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  minmod(const real_t a, const real_t b, const real_t c, const real_t Mdx2_) const
  {
    real_t aa = fabs(a);
    if (aa < Mdx2_)
      return a;

    if (a * b > 0 && b * c > 0)
    {
      real_t s = (a > 0) ? 1.0 : -1.0;
      return s * fmin(aa, fmin(fabs(b), fabs(c)));
    }
    else
      return 0;

  } // minmod


  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 2, int>::type & index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    const real_t gamma0 = this->params.settings.gamma0;

    HydroState DuX;
    HydroState DuX_new;
    HydroState DuXL; // neighbor on the left
    HydroState DuXR; // neighbor on the right

    HydroState DuY;
    HydroState DuY_new;
    HydroState DuYL; // neighbor on the left
    HydroState DuYR; // neighbor on the right

    // Needs explanation here !!!
    const real_t dx = 1.0; // this->params.dx;

    // local cell index
    int i, j;
    index2coord(index, i, j, isize, jsize);

    if (i == 0 or i == isize - 1 or j == 0 or j == jsize - 1)
      return;

    // read cell-average conservative variables
    HydroState Uave = {}; // Uave[IE]=0;
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {
      Uave[ivar] = Uaverage(i, j, ivar);
    }

    // speed of sound from cell-averaged values
    const real_t c = euler.compute_speed_of_sound(Uave, gamma0);

    // read cell-averaged gradient, and compute difference with close neighbors
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      DuX[ivar] = Ugradx(i, j, ivar) * dx;
      DuXL[ivar] = Uaverage(i, j, ivar) - Uaverage(i - 1, j, ivar);
      DuXR[ivar] = Uaverage(i + 1, j, ivar) - Uaverage(i, j, ivar);

      DuY[ivar] = Ugrady(i, j, ivar) * dx;
      DuYL[ivar] = Uaverage(i, j, ivar) - Uaverage(i, j - 1, ivar);
      DuYR[ivar] = Uaverage(i, j + 1, ivar) - Uaverage(i, j, ivar);
    }

    // if limiter_characteristics_enabled ...
    // transform to characteristics variables
    euler.template cons_to_charac<IX>(DuX, Uave, c, gamma0);
    euler.template cons_to_charac<IX>(DuXL, Uave, c, gamma0);
    euler.template cons_to_charac<IX>(DuXR, Uave, c, gamma0);

    euler.template cons_to_charac<IY>(DuY, Uave, c, gamma0);
    euler.template cons_to_charac<IY>(DuYL, Uave, c, gamma0);
    euler.template cons_to_charac<IY>(DuYR, Uave, c, gamma0);

    // Apply minmod limiter
    double       change_x = 0;
    double       change_y = 0;
    const double beta = 1.0;

    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      DuX_new[ivar] = minmod(DuX[ivar], beta * DuXL[ivar], beta * DuXR[ivar], Mdx2);
      change_x += fabs(DuX_new[ivar] - DuX[ivar]);

      DuY_new[ivar] = minmod(DuY[ivar], beta * DuYL[ivar], beta * DuYR[ivar], Mdx2);
      change_y += fabs(DuY_new[ivar] - DuY[ivar]);
    }
    change_x /= nbvar;
    change_y /= nbvar;


    // If limiter is active, reduce polynomial to linear
    // recompute all DoF in current cell
    if (change_x + change_y > 1.0e-10)
    {

      for (int ivar = 0; ivar < nbvar; ++ivar)
      {
        DuX_new[ivar] /= dx;
        DuY_new[ivar] /= dx;
      }

      euler.template charac_to_cons<IX>(DuX_new, Uave, c, gamma0);
      euler.template charac_to_cons<IY>(DuY_new, Uave, c, gamma0);

      // for each variable : ID, IE, IU, IV
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // for each dof
        for (int idy = 0; idy < N; ++idy)
        {

          real_t ry = this->sdm_geom.solution_pts_1d(idy) - 0.5;

          for (int idx = 0; idx < N; ++idx)
          {

            real_t rx = this->sdm_geom.solution_pts_1d(idx) - 0.5;

            Udata(i, j, dofMap(idx, idy, 0, ivar)) =
              Uave[ivar] + rx * DuX_new[ivar] + ry * DuY_new[ivar];

          } // end for idx

        } // end for idy

      } // end for ivar

    } // end change_x + change_y

  } // operator () - 2d

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d
  template <int dim_ = dim>
  KOKKOS_INLINE_FUNCTION void
  operator()(const typename std::enable_if<dim_ == 3, int>::type & index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    const real_t gamma0 = this->params.settings.gamma0;

    HydroState DuX;
    HydroState DuX_new;
    HydroState DuXL; // neighbor on the left
    HydroState DuXR; // neighbor on the right

    HydroState DuY;
    HydroState DuY_new;
    HydroState DuYL; // neighbor on the left
    HydroState DuYR; // neighbor on the right

    HydroState DuZ;
    HydroState DuZ_new;
    HydroState DuZL; // neighbor on the left
    HydroState DuZR; // neighbor on the right

    // Needs explanation here !!!
    const real_t dx = 1.0; // this->params.dx;

    // local cell index
    int i, j, k;
    index2coord(index, i, j, k, isize, jsize, ksize);

    if (i == 0 or i == isize - 1 or j == 0 or j == jsize - 1 or k == 0 or k == ksize - 1)
      return;

    // read cell-average conservative variables
    HydroState Uave = {}; // Uave[IE]=0;
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {
      Uave[ivar] = Uaverage(i, j, k, ivar);
    }

    // speed of sound from cell-averaged values
    const real_t c = euler.compute_speed_of_sound(Uave, gamma0);

    // read cell-averaged gradient, and compute difference with close neighbors
    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      DuX[ivar] = Ugradx(i, j, k, ivar) * dx;
      DuXL[ivar] = Uaverage(i, j, k, ivar) - Uaverage(i - 1, j, k, ivar);
      DuXR[ivar] = Uaverage(i + 1, j, k, ivar) - Uaverage(i, j, k, ivar);

      DuY[ivar] = Ugrady(i, j, k, ivar) * dx;
      DuYL[ivar] = Uaverage(i, j, k, ivar) - Uaverage(i, j - 1, k, ivar);
      DuYR[ivar] = Uaverage(i, j + 1, k, ivar) - Uaverage(i, j, k, ivar);

      DuZ[ivar] = Ugradz(i, j, k, ivar) * dx;
      DuZL[ivar] = Uaverage(i, j, k, ivar) - Uaverage(i, j, k - 1, ivar);
      DuZR[ivar] = Uaverage(i, j, k + 1, ivar) - Uaverage(i, j, k, ivar);
    }

    // if limiter_characteristics_enabled ...
    // transform to characteristics variables
    euler.template cons_to_charac<IX>(DuX, Uave, c, gamma0);
    euler.template cons_to_charac<IX>(DuXL, Uave, c, gamma0);
    euler.template cons_to_charac<IX>(DuXR, Uave, c, gamma0);

    euler.template cons_to_charac<IY>(DuY, Uave, c, gamma0);
    euler.template cons_to_charac<IY>(DuYL, Uave, c, gamma0);
    euler.template cons_to_charac<IY>(DuYR, Uave, c, gamma0);

    euler.template cons_to_charac<IZ>(DuZ, Uave, c, gamma0);
    euler.template cons_to_charac<IZ>(DuZL, Uave, c, gamma0);
    euler.template cons_to_charac<IZ>(DuZR, Uave, c, gamma0);

    // Apply minmod limiter
    double       change_x = 0;
    double       change_y = 0;
    double       change_z = 0;
    const double beta = 1.0;

    for (int ivar = 0; ivar < nbvar; ++ivar)
    {

      DuX_new[ivar] = minmod(DuX[ivar], beta * DuXL[ivar], beta * DuXR[ivar], Mdx2);
      change_x += fabs(DuX_new[ivar] - DuX[ivar]);

      DuY_new[ivar] = minmod(DuY[ivar], beta * DuYL[ivar], beta * DuYR[ivar], Mdx2);
      change_y += fabs(DuY_new[ivar] - DuY[ivar]);

      DuZ_new[ivar] = minmod(DuZ[ivar], beta * DuZL[ivar], beta * DuZR[ivar], Mdx2);
      change_z += fabs(DuZ_new[ivar] - DuZ[ivar]);
    }
    change_x /= nbvar;
    change_y /= nbvar;
    change_z /= nbvar;

    // If limiter is active, reduce polynomial to linear
    // recompute all DoF in current cell
    if (change_x + change_y + change_z > 1.0e-10)
    {

      for (int ivar = 0; ivar < nbvar; ++ivar)
      {
        DuX_new[ivar] /= dx;
        DuY_new[ivar] /= dx;
        DuZ_new[ivar] /= dx;
      }

      euler.template charac_to_cons<IX>(DuX_new, Uave, c, gamma0);
      euler.template charac_to_cons<IY>(DuY_new, Uave, c, gamma0);
      euler.template charac_to_cons<IZ>(DuZ_new, Uave, c, gamma0);

      // for each variable : ID, IE, IU, IV, IW
      for (int ivar = 0; ivar < nbvar; ++ivar)
      {

        // for each dof
        for (int idz = 0; idz < N; ++idz)
        {

          real_t rz = this->sdm_geom.solution_pts_1d(idz) - 0.5;

          for (int idy = 0; idy < N; ++idy)
          {

            real_t ry = this->sdm_geom.solution_pts_1d(idy) - 0.5;

            for (int idx = 0; idx < N; ++idx)
            {

              real_t rx = this->sdm_geom.solution_pts_1d(idx) - 0.5;

              Udata(i, j, k, dofMap(idx, idy, idz, ivar)) =
                Uave[ivar] + rx * DuX_new[ivar] + ry * DuY_new[ivar] + rz * DuZ_new[ivar];

            } // end for idx

          } // end for idy

        } // end for idz

      } // end for ivar

    } // end change_x + change_y + change_z

  } // operator () - 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray                   Udata;
  DataArray                   Uaverage;
  DataArray                   Ugradx;
  DataArray                   Ugrady;
  DataArray                   Ugradz;
  real_t                      Mdx2;

}; // class Apply_limiter_Functor

} // namespace sdm
} // namespace ppkMHD

#endif // SDM_LIMITER_FUNCTORS_H_
