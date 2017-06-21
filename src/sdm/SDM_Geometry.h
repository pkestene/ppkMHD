#ifndef SDM_GEOMETRY_H_
#define SDM_GEOMETRY_H_

#include "kokkos_shared.h"

namespace sdm {

/**
 * enumerate all possible locations for the solution points, specified as
 * a quadrature.
 */
enum SDM_SOLUTION_POINTS_TYPE {
  
  SDM_SOL_GAUSS_CHEBYSHEV = 0
  
}; // enum SDM_SOLUTION_POINTS_TYPE
  
/**
 * enumerate all possible locations for the solution points, specified as
 * a quadrature.
 */
enum SDM_FLUX_POINTS_TYPE {
  
  SDM_FLUX_GAUSS_LEGENDRE = 0,
  SDM_FLUX_GAUSS_CHEBYSHEV = 1,
  
}; // enum SDM_FLUX_POINTS_TYPE

/**
 * \class SDM_Geometry
 *
 * This class stores data array (Kokkos::View) to hold the location of:
 *
 * - solution points location (Gauss-Chebyshev quadrature points)
 * - flux points (Gauss-Legendre or Chebyshev-Gauss-Lobatto)
 *
 * \tparam dim is dimension of space (2 or 3).
 *
 * In this code, we are only interested in 2D/3D hexahedral cells.
 */
template <int dim>
class SDM_Geometry
{

public:
  using PointsArray1D = Kokkos::View<real_t*, DEVICE>;
  using PointsArray   = Kokkos::View<real_t*[dim], DEVICE>;

  using PointsArray1DHost = PointsArray1D::HostMirror;
  using PointsArrayHost   = PointsArray::HostMirror;
  
  SDM_Geometry();
  ~SDM_Geometry();

  PointsArray1D solution_points_1d;
  PointsArray   solution_points;

  PointsArray1D flux_points_1d;
  PointsArray   flux_points_x;
  PointsArray   flux_points_y;
  PointsArray   flux_points_z;
  
  /**
   * Init SDM_Geometry class.
   *
   * \param[in] N is the order of the scheme should be in [1,5].
   * \param[in] sdm_sol_pts_type is the type of quadrature used for solution points
   * \param[in] sdm_flux_pts_type is the type of quadrature used for fluxes points
   */
  void init(int N,
	    SDM_SOLUTION_POINTS_TYPE sdm_sol_pts_type,
	    SDM_FLUX_POINTS_TYPE     sdm_flux_pts_type)
  {
    
    solution_pts_1d = PointsArray1D("solution_pts_1d",N);
    PointsArray1DHost solution_pts_1d_host = Kokkos::create_mirror(solution_pts_1d);
    
    // init 1d solution points
    if (sdm_sol_pts_type == SDM_SOL_GAUSS_CHEBYSHEV) {
      
      //for (int i=1; i<N; i++)
      //  solution_pts_1d_host(i-1) = 0.5 * (1 - cos(M_PI*(2*i-1)/2/N));
      for (int i=0; i<N-1; i++)
	solution_pts_1d_host(i) = 0.5 * (1 - cos(M_PI*(2*i+1)/2/N));
      
    }
    
    // number of solution points
    int N_solution_pts = N*N;
    if (dim==3)
      N_solution_pts = N*N*N;
    
    // create tensor product solution points locations
    solution_pts_host = PointsArrayHost("solution_pts_host",N_solution_pts);

    if (dim == 2) {
    } else {
    }
    
  } // init

}; // class SDM_Geometry

} // namespace sdm

#endif // SDM_GEOMETRY_H_
