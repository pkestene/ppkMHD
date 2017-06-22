#ifndef SDM_GEOMETRY_H_
#define SDM_GEOMETRY_H_

#include "shared/kokkos_shared.h"

namespace sdm {

/**
 * enumerate all possible locations for the solution points, specified as
 * a quadrature.
 */
enum SDM_SOLUTION_POINTS_TYPE {
  
  SDM_SOL_GAUSS_CHEBYSHEV = 0
  
}; // enum SDM_SOLUTION_POINTS_TYPE
  
/**
 * enumerate all possible locations for the flux points, specified as
 * a quadrature.
 */
enum SDM_FLUX_POINTS_TYPE {
  
  SDM_FLUX_GAUSS_LEGENDRE = 0, // roots of P_{n-1} + the 2 end points
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
  //! for 1D quadrature location
  using PointsArray1D = Kokkos::View<real_t*, DEVICE>;

  //! for 2D solution/flux location
  using PointsArray2D = Kokkos::View<real_t**[2], DEVICE>;

  //! for 3D solution/flux location
  using PointsArray3D = Kokkos::View<real_t***[3], DEVICE>;

  //! generic alias
  using PointsArray   = typename std::conditional<dim==2,PointsArray2D,PointsArray3D>::type;

  //! host mirror aliases
  using PointsArray1DHost = PointsArray1D::HostMirror;
  using PointsArray2DHost = PointsArray2D::HostMirror;
  using PointsArray3DHost = PointsArray3D::HostMirror;
  using PointsArrayHost   = typename PointsArray::HostMirror;
  
  SDM_Geometry() {};
  ~SDM_Geometry() {};

  PointsArray1D     solution_pts_1d;
  PointsArray1DHost solution_pts_1d_host;
  PointsArray       solution_pts;

  PointsArray1D     flux_pts_1d;
  PointsArray1DHost flux_pts_1d_host;
  PointsArray   flux_x_pts;
  PointsArray   flux_y_pts;
  PointsArray   flux_z_pts;
  
  /**
   * Init solution and flux points locations.
   *
   * \param[in] N is the order of the scheme
   *
   * There are N solution points per dimension
   * and N+1 flux points per dimension.
   *
   * This is where solution_pts
   * and flux_x_pts, flux_y_pts, flux_z_pts are allocated.
   */
  void init(int N);
  
private:

  /**
   * Init SDM_Geometry class 1d point locations.
   *
   * Solution points and flux points coordinates in reference cell 
   * in units [0,1]^dim
   *
   * \param[in] N is the order of the scheme should be in [1,5].
   * \param[in] sdm_sol_pts_type is the type of quadrature used for solution points
   * \param[in] sdm_flux_pts_type is the type of quadrature used for fluxes points
   */
  void init_1d(int N,
	       SDM_SOLUTION_POINTS_TYPE sdm_sol_pts_type,
	       SDM_FLUX_POINTS_TYPE     sdm_flux_pts_type)
  {

    // =================
    // Solution points
    // =================
    
    // memory allocate solution_pts_1d
    solution_pts_1d = PointsArray1D("solution_pts_1d",N);
    solution_pts_1d_host = Kokkos::create_mirror(solution_pts_1d);
    
    // init 1d solution points
    if (sdm_sol_pts_type == SDM_SOL_GAUSS_CHEBYSHEV) {
      
      //for (int i=1; i<N; i++)
      //  solution_pts_1d_host(i-1) = 0.5 * (1 - cos(M_PI*(2*i-1)/2/N));
      for (int i=0; i<N-1; i++)
	solution_pts_1d_host(i) = 0.5 * (1 - cos(M_PI*(2*i+1)/2/N));
      
    }

    // copy to device
    Kokkos::deep_copy(solution_pts_1d, solution_pts_1d_host);
    
    // =================
    // Flux points
    // =================

    // memory allocate flux_pts_1d
    flux_pts_1d = PointsArray1D("flux_pts_1d",N+1);
    flux_pts_1d_host = Kokkos::create_mirror(flux_pts_1d);

    // init 1d flux points
    if (sdm_flux_pts_type == SDM_FLUX_GAUSS_LEGENDRE) {

      // use the roots of Legendre polynomial of degree N-1 + the two end points
      if (N==1) {        // use Legendre P_0 = 1

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = 1.0;

      } else if (N==2) { // use Legendre P_1 = x

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = 0.5;
	flux_pts_1d_host(2) = 1.0;
	
      } else if (N==3) { // use Legendre P_2 = 0.5*(3x^2-1)

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = (1-1.0/sqrt(3.0))/2.0;
	flux_pts_1d_host(2) = (1+1.0/sqrt(3.0))/2.0;
	flux_pts_1d_host(3) = 1.0;
	
      } else if (N==4) { // use Legendre P_3 = 0.5*(5x^3-3x)

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = 0.5 * (1-sqrt(3.0/5));
	flux_pts_1d_host(2) = 0.5;
	flux_pts_1d_host(3) = 0.5 * (1+sqrt(3.0/5));
	flux_pts_1d_host(4) = 1.0;
	
      } else if (N==5) { // use Legendre P_4 = 1/8*(35x^4-30x^2+3)

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = 0.5 * ( 1.0 - sqrt(3.0/7 + 2.0/7*sqrt(6.0/5)) );
	flux_pts_1d_host(2) = 0.5 * ( 1.0 - sqrt(3.0/7 - 2.0/7*sqrt(6.0/5)) );
	flux_pts_1d_host(3) = 0.5 * ( 1.0 + sqrt(3.0/7 - 2.0/7*sqrt(6.0/5)) );
	flux_pts_1d_host(4) = 0.5 * ( 1.0 + sqrt(3.0/7 + 2.0/7*sqrt(6.0/5)) );
	flux_pts_1d_host(5) = 1.0;
	
      } else if (N==6) { // use Legendre P_5 = 1/8*(63x^5-70x^3+15x)

	flux_pts_1d_host(0) = 0.0;
	flux_pts_1d_host(1) = 0.5 * ( 1.0 - 1.0/3*sqrt(5.0 + 2.0*sqrt(10.0/7)) ) ;
	flux_pts_1d_host(2) = 0.5 * ( 1.0 - 1.0/3*sqrt(5.0 - 2.0*sqrt(10.0/7)) ) ;
	flux_pts_1d_host(3) = 0.5;
	flux_pts_1d_host(4) = 0.5 * ( 1.0 + 1.0/3*sqrt(5.0 - 2.0*sqrt(10.0/7)) ) ;
	flux_pts_1d_host(5) = 0.5 * ( 1.0 + 1.0/3*sqrt(5.0 + 2.0*sqrt(10.0/7)) ) ;
	flux_pts_1d_host(6) = 1.0;
	
      }
    }

    // copy to device
    Kokkos::deep_copy(flux_pts_1d, flux_pts_1d_host);

  } // init_1d

}; // class SDM_Geometry

// =======================================================
// =======================================================
template<>
void SDM_Geometry<2>::init(int N)
{

  // first 1d initialization
  init_1d(N, SDM_SOL_GAUSS_CHEBYSHEV, SDM_FLUX_GAUSS_LEGENDRE);

  // =================
  // Solution points
  // =================
  
  // perform tensor product for solution points

  // create tensor product solution points locations
  solution_pts = PointsArray("solution_pts",N,N);
  PointsArrayHost solution_pts_host = Kokkos::create_mirror(solution_pts);
  
  for (int j=0; j<N; ++j) {
    for (int i=0; i<N; ++i) {
      solution_pts_host(i,j,0) = solution_pts_1d_host(i);
      solution_pts_host(i,j,1) = solution_pts_1d_host(j);
    }
  }
  
  // copy solution point coordinates on DEVICE
  Kokkos::deep_copy(solution_pts, solution_pts_host);

  // =================
  // Flux points
  // =================
  
  flux_x_pts = PointsArray("flux_x_pts",N+1, N  );
  flux_y_pts = PointsArray("flux_y_pts",N  , N+1);

  PointsArrayHost flux_x_pts_host = Kokkos::create_mirror(flux_x_pts);
  PointsArrayHost flux_y_pts_host = Kokkos::create_mirror(flux_y_pts);

  // flux_x located at the same y-coordinates as the solution points
  for (int j=0; j<N; ++j) {
    for (int i=0; i<N+1; ++i) {
      flux_x_pts_host(i,j,0) = flux_pts_1d_host(i);
      flux_x_pts_host(i,j,1) = solution_pts_1d_host(j);
    }
  }

  // flux_y located at the same x-coordinates as the solution points
  for (int j=0; j<N+1; ++j) {
    for (int i=0; i<N; ++i) {
      flux_y_pts_host(i,j,0) = solution_pts_1d_host(i);
      flux_y_pts_host(i,j,1) = flux_pts_1d_host(j);
    }
  }
  
} // SDM_Geometry::init<2>

// =======================================================
// =======================================================
template<>
void SDM_Geometry<3>::init(int N)
{

  // first 1d initialization
  init_1d(N, SDM_SOL_GAUSS_CHEBYSHEV, SDM_FLUX_GAUSS_LEGENDRE);

  // =================
  // Solution points
  // =================

  // perform tensor product for solution points

  // create tensor product solution points locations
  solution_pts = PointsArray("solution_pts",N,N,N);
  PointsArrayHost solution_pts_host = Kokkos::create_mirror(solution_pts);

  for (int k=0; k<N; ++k) {
    for (int j=0; j<N; ++j) {
      for (int i=0; i<N; ++i) {
	solution_pts_host(i,j,k,0) = solution_pts_1d_host(i);
	solution_pts_host(i,j,k,1) = solution_pts_1d_host(j);
	solution_pts_host(i,j,k,2) = solution_pts_1d_host(k);
      }
    }
  }

  // copy solution point coordinates on DEVICE
  Kokkos::deep_copy(solution_pts, solution_pts_host);
  
  // =================
  // Flux points
  // =================
  
  flux_x_pts = PointsArray("flux_x_pts",N+1, N  , N  );
  flux_y_pts = PointsArray("flux_y_pts",N  , N+1, N  );
  flux_z_pts = PointsArray("flux_z_pts",N  , N  , N+1);

  PointsArrayHost flux_x_pts_host = Kokkos::create_mirror(flux_x_pts);
  PointsArrayHost flux_y_pts_host = Kokkos::create_mirror(flux_y_pts);
  PointsArrayHost flux_z_pts_host = Kokkos::create_mirror(flux_z_pts);

  // flux_x located at the same y,z-coordinates as the solution points
  for (int k=0; k<N; ++k) {
    for (int j=0; j<N; ++j) {
      for (int i=0; i<N+1; ++i) {
	flux_x_pts_host(i,j,k,0) = flux_pts_1d_host(i);
	flux_x_pts_host(i,j,k,1) = solution_pts_1d_host(j);
	flux_x_pts_host(i,j,k,2) = solution_pts_1d_host(k);
      }
    }
  }

  // flux_y located at the same x,z-coordinates as the solution points
  for (int k=0; k<N; ++k) {
    for (int j=0; j<N+1; ++j) {
      for (int i=0; i<N; ++i) {
	flux_y_pts_host(i,j,k,0) = solution_pts_1d_host(i);
	flux_y_pts_host(i,j,k,1) = flux_pts_1d_host(j);
	flux_y_pts_host(i,j,k,2) = solution_pts_1d_host(k);
      }
    }
  }

  // flux_z located at the same y,z-coordinates as the solution points
  for (int k=0; k<N+1; ++k) {
    for (int j=0; j<N; ++j) {
      for (int i=0; i<N; ++i) {
	flux_z_pts_host(i,j,k,0) = solution_pts_1d_host(i);
	flux_z_pts_host(i,j,k,1) = solution_pts_1d_host(j);
	flux_z_pts_host(i,j,k,2) = flux_pts_1d_host(k);
      }
    }
  }

} // SDM_Geometry::init<3>


} // namespace sdm

#endif // SDM_GEOMETRY_H_
