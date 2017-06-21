#ifndef SDM_GEOMETRY_H_
#define SDM_GEOMETRY_H_

#include "kokkos_shared.h"

namespace sdm {

/**
 * \class SDM_Geometry
 *
 * This class stores data array (Kokkos::View) to hold the location of:
 *
 * - solution points location (Gauss-Chebyshev quadrature points)
 * - flux points (Gauss-Legendre or Chebyshev-Gauss-Lobatto)
 *
 * In this code, we are only interested in 2D/3D hexahedral cells
 */
template <int dim>
class SDM_Geometry
{

public:
  using PointsArray1D = Kokkos::View<real_t*, DEVICE>;
  using PointsArray   = Kokkos::View<real_t*[dim], DEVICE>;
  
  SDM_Geometry();
  ~SDM_Geometry();

  PointsArray1D solution_points_1d;
  PointsArray   solution_points;

  PointsArray1D flux_points_1d;
  PointsArray   flux_points_x;
  PointsArray   flux_points_y;
  PointsArray   flux_points_z;
  
}; // class SDM_Geometry

/**
 * Init SDM_Geometry class.
 */

} // namespace sdm

#endif // SDM_GEOMETRY_H_
