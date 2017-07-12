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
  
  SDM_FLUX_GAUSS_LEGENDRE = 0, // roots of P_{n-1} + the 2 end points 0 and 1
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
 * This class also holds some Lagrange polynomial matrix, because we
 * want to interpolate at flux points using the Lagrange polynomial basis 
 * at solution points, and conversely interpolate at solution points using 
 * Lagrange polynomial basis at flux points.
 * 
 * Interpolation is done direction by direction, so we only need 1D Lagrange basis.
 * When using Lagrange points at solution points, the Lagrange matrix is N by N+1, 
 * made of N+1 columns (one for each interpolated flux point) of length N (because
 * there is exactly N different Lagrange polynomials, one for each solution point).
 * 
 *
 * \tparam dim is dimension of space (2 or 3).
 * \tparam order is the scheme order, also equal to N (number of solution points per dim).
 * order should be in [1,6]
 *
 *
 * In this code, we are only interested in 2D/3D hexahedral cells.
 */
template <int dim,
	  int order>
class SDM_Geometry
{

public:

  /**
   * Number of solution points (per direction).
   *
   * Notice that the total number of solution points is N^d in dimension dim.
   */
  static const int N = order;
  
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

  //! Lagrange interpolation matrix type
  using LagrangeMatrix     = Kokkos::View<real_t **, DEVICE>;
  using LagrangeMatrixHost = LagrangeMatrix::HostMirror;

  SDM_Geometry() {};
  ~SDM_Geometry() {};

  /**
   * \defgroup location-point arrays
   */
  /**@{*/

  //! location of solution points in reference cell [0,1]^dim, device array
  PointsArray1D     solution_pts_1d;

  //! location of solution points in reference cell [0,1]^dim, host array
  PointsArray1DHost solution_pts_1d_host;

  //! dim-dimensional array of solution points location (obtained by tensorial product)
  PointsArray       solution_pts;

  //! dim-dimensional array of solution points location on host (for IO routines)
  PointsArrayHost   solution_pts_host;
  
  //! location of flux points in reference cell [0,1]^dim, device array
  PointsArray1D     flux_pts_1d;

  //! location of flux points in reference cell [0,1]^dim, host array
  PointsArray1DHost flux_pts_1d_host;

  //! dim-dimensional array of flux points location (obtained by tensorial product)
  PointsArray   flux_x_pts;
  PointsArray   flux_y_pts;
  PointsArray   flux_z_pts;
  /**@}*/

  /**
   * \defgroup Lagrange-interpolation
   */
  /**@{*/

  //! Lagrange matrix to interpolate at flux points
  //! N   lines : one basis element per solution points
  //! N+1 cols  : one per interpolated point (flux points)
  //! sol2flux(i,j) is the value of the i-th Lagrange polynomial (i-th
  //! solution point) taken at the j-th flux point.
  LagrangeMatrix sol2flux;

  //! Lagrange matrix to interpolate at solution points
  //! flux2sol matrix has
  //! N+1 lines : one basis element per flux points
  //! N   cols  : one per interpolated point (solution points)
  //! flux2sol(i,j) is the value of the i-th Lagrange polynomial (i-th
  //! flux point) taken at the i-th solution point.
  LagrangeMatrix flux2sol;

  //! Lagrange matrix to interpolate flux derivative at solution points
  //! flux2sol_derivatives matrix has
  //! N+1 lines : one basis element per flux points
  //! N   cols  : one per interpolated point (solution points)
  //! flux2sol_derivative(i,j) is the value of the i-th Lagrange polynomial derivative
  //! (i-th flux point) taken at the i-th solution point.
  LagrangeMatrix flux2sol_derivative;

  /**@}*/

  
  /**
   * Init solution and flux points locations.
   *
   * There are N solution points per dimension
   * and N+1 flux points per dimension.
   *
   * This is where solution_pts
   * and flux_x_pts, flux_y_pts, flux_z_pts are allocated.
   */
  // =======================================================
  // ======================================================= 
  template<int dim_ = dim>
  void init(const typename std::enable_if<dim_==2, int>::type& dummy)
  {
    
    // first 1d initialization
    init_1d(SDM_SOL_GAUSS_CHEBYSHEV, SDM_FLUX_GAUSS_LEGENDRE);
    
    // =================
    // Solution points
    // =================
    
    // perform tensor product for solution points
    
    // create tensor product solution points locations
    solution_pts = PointsArray("solution_pts",N,N);
    solution_pts_host = Kokkos::create_mirror(solution_pts);
    
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
    
  } // init for 2D

  // ======================================================= 
  // =======================================================
  template<int dim_ = dim>
  void init(const typename std::enable_if<dim_==3, int>::type& dummy)
  {
    
    // first 1d initialization
    init_1d(SDM_SOL_GAUSS_CHEBYSHEV, SDM_FLUX_GAUSS_LEGENDRE);
    
    // =================
    // Solution points
    // =================
    
    // perform tensor product for solution points
    
    // create tensor product solution points locations
    solution_pts = PointsArray("solution_pts",N,N,N);
    solution_pts_host = Kokkos::create_mirror(solution_pts);
    
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
    
  } // init for 3D
  
private:

  /**
   * Init 1d point locations (solution + flux).
   *
   * Solution points and flux points coordinates in reference cell 
   * in units [0,1]^dim
   *
   * \param[in] sdm_sol_pts_type is the type of quadrature used for solution points
   * \param[in] sdm_flux_pts_type is the type of quadrature used for fluxes points
   */
  void init_1d(SDM_SOLUTION_POINTS_TYPE sdm_sol_pts_type,
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
      
      //for (int i=1; i<=N; i++)
      //  solution_pts_1d_host(i-1) = 0.5 * (1 - cos(M_PI*(2*i-1)/2/N));
      for (int i=0; i<N; i++)
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

public:
  /**
   * Create Lagrange interpolation matrix.
   */
  void init_lagrange_1d();

}; // class SDM_Geometry

// =======================================================
// =======================================================
template<int dim,
	 int order>
void SDM_Geometry<dim,order>::init_lagrange_1d()
{
  
  // memory allocation

  /////////////
  //
  // sol2flux
  //
  ////////////
  // N   lines : one basis element per solution points
  // N+1 cols  : one per interpolated point (flux points)
  sol2flux = LagrangeMatrix("sol2flux",N,N+1);

  LagrangeMatrixHost sol2flux_h = Kokkos::create_mirror(sol2flux);

  // create i,j entries in Lagrange matrix
  // i is i-th Lagrange polynomial (solution)
  // j is the location of interpolated point (flux)
  for (int j=0; j<N+1; ++j) {

    real_t x_j = flux_pts_1d_host(j);
    
    for (int i=0; i<N; ++i) {

      real_t x_i = solution_pts_1d_host(i);
      
      /*
       * Lagrange polynomial (solution points basis)
       *
       * l_i(x) = \Pi_{k \neq i} \frac{x-x_k}{x_i-x_k}
       */
      real_t l = 1.0;

      // k spans Lagrange basis, number of solution points
      for (int k=0; k<N; ++k) {
	real_t x_k = solution_pts_1d_host(k);
	if (k != i) {
	  l *= (x_j-x_k)/(x_i-x_k);
	}
      }

      // copy l into matrix
      sol2flux_h(i,j) = l;
      
    } // end for i

  } // end for j

  Kokkos::deep_copy(sol2flux,sol2flux_h);

  
  /////////////
  //
  // flux2sol
  //
  /////////////
  // N+1 lines : one basis element per flux points
  // N   cols  : one per interpolated point (solution points)
  flux2sol = LagrangeMatrix("flux2sol",N+1,N);
  
  LagrangeMatrixHost flux2sol_h = Kokkos::create_mirror(flux2sol);

  // create i,j entries in Lagrange matrix flux2sol
  // i is i-th Lagrange polynomial (flux)
  // j is the location of interpolated point (solution)
  for (int j=0; j<N; ++j) {

    real_t x_j = solution_pts_1d_host(j);
    
    for (int i=0; i<N+1; ++i) {

      real_t x_i = flux_pts_1d_host(i);
      
      /*
       * Lagrange polynomial (flux points basis)
       *
       * l_i(x) = \Pi_{k \neq i} \frac{x-x_k}{x_i-x_k}
       */
      real_t l = 1.0;

      // k spans Lagrange basis, number of flux points
      for (int k=0; k<N+1; ++k) {
	real_t x_k = flux_pts_1d_host(k);
	if (k != i) {
	  l *= (x_j-x_k)/(x_i-x_k);
	}
      }

      // copy l into matrix
      flux2sol_h(i,j) = l;
      
    } // end for i

  } // end for j

  Kokkos::deep_copy(flux2sol,flux2sol_h);
  
  ////////////////////////
  //
  // flux2sol_derivative
  //
  ///////////////////////
  // N+1 lines : one basis element per flux points
  // N   cols  : one per interpolated point (solution points)
  flux2sol_derivative = LagrangeMatrix("flux2sol_derivative",N+1,N);
  
  LagrangeMatrixHost flux2sol_derivative_h = Kokkos::create_mirror(flux2sol_derivative);

  // create i,j entries in Lagrange matrix flux2sol_derivative
  // i is i-th Lagrange polynomial derivative (flux)
  // j is the location of interpolated point (solution)
  Kokkos::deep_copy(flux2sol_derivative,flux2sol_derivative_h);
  
} // SDM_Geometry::init_lagrange_1d

} // namespace sdm

#endif // SDM_GEOMETRY_H_
