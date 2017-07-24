/**
 * Testing ideas for least-squares estimation of gradient at cell center,
 * using data at solution nodes.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "sdm/SDM_Geometry.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

//! a test scalar function
real_t f(real_t x, real_t y, real_t z)
{
  return 2.23*x*x + x*5 -2.2*x*y + 5*x*z*z + 3 + y*z + 1/(x+2+y*y);
}

//! partial derivative along x of the test function
real_t f_x(real_t x, real_t y, real_t z)
{
  return 2.23*2*x + 5 -2.2*y + 5*z*z - 1.0/(x+2+y*y)/(x+2+y*y);
} // f_x

//! partial derivative along x of the test function
real_t f_y(real_t x, real_t y, real_t z)
{
  return -2.2*x + z - 2*y/(x+2+y*y)/(x+2+y*y);
} // f_y

//! partial derivative along x of the test function
real_t f_z(real_t x, real_t y, real_t z)
{
  return 5*x*2*z + y;
} // f_z

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
template<int dim,
	 int N>
void test_sdm_lsq_gradient()
{

  std::cout << "\n\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "  Dimension is : " << dim << "\n";
  std::cout << "  Using order : "  << N   << "\n";
  std::cout << "  Number of solution points : " << N << "\n";
  std::cout << "  Number of flux     points : " << N+1 << "\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  std::cout << "===============================================\n";
  
  sdm::SDM_Geometry<dim,N> sdm_geom;
  sdm_geom.init(0);

  // matrix elements
  real_t mat[dim][dim] = {0};

  // right hand side
  real_t rhs[dim] = {0};
  
  // center
  real_t xc = 0.5;
  real_t yc = 0.5;
  real_t zc = 0.5;
  UNUSED(zc);
  
  if (dim==2) {

    real_t z = 1.0;
    
    for (int j=0; j<N; ++j) {
      real_t y = sdm_geom.solution_pts_1d_host(j);
      
      for (int i=0; i<N; ++i) {
	real_t x = sdm_geom.solution_pts_1d_host(i);

	real_t dx = x-xc;
	real_t dy = y-yc;
	
	//real_t w = 1.0/sqrt( dx*dx + dy*dy );
	real_t w = 1.0;
	
	mat[0][0] += w*dx*dx;
	mat[0][1] += w*dx*dy;
	mat[1][1] += w*dy*dy;

	real_t df = f(x,y,z);// - f(xc,yc,z);
	rhs[0] += w*dx*df;
	rhs[1] += w*dy*df;
	
      } // end i
    } // end j

    // symetry
    mat[1][0] = mat[0][1];

    printf("%f %f\n",mat[0][0], mat[0][1]);
    printf("%f %f\n",mat[1][0], mat[1][1]);

    printf("df/dx=%f - exact is %f\n",rhs[0]/mat[0][0],f_x(xc,yc,z));
    printf("df/dy=%f - exact is %f\n",rhs[1]/mat[1][1],f_y(xc,yc,z));
    
  } else {

    // compute only diagonal terms
    for (int k=0; k<N; ++k) {
      real_t z = sdm_geom.solution_pts_1d_host(k);

      for (int j=0; j<N; ++j) {
	real_t y = sdm_geom.solution_pts_1d_host(j);
	
	for (int i=0; i<N; ++i) {
	  real_t x = sdm_geom.solution_pts_1d_host(i);

	  real_t dx = x-xc;
	  real_t dy = y-yc;
	  real_t dz = z-zc;
	  
	  //real_t w = 1.0/sqrt( dx*dx + dy*dy );
	  real_t w = 1.0;
	
	  mat[0][0] += w*dx*dx;
	  mat[1][1] += w*dy*dy;
	  mat[2][2] += w*dz*dz;

	  real_t df = f(x,y,z);// - f(xc,yc,z);
	  rhs[0] += w*dx*df;
	  rhs[1] += w*dy*df;
	  rhs[2] += w*dz*df;
	
	} // end i
      } // end j
    } // end k

    printf("%f %f %f\n",mat[0][0], mat[1][1], mat[2][2]);

    /*printf("df/dx=%f - exact is %f\n",rhs[0]/mat[0][0],f_x(xc,yc,zc));
      printf("df/dy=%f - exact is %f\n",rhs[1]/mat[1][1],f_y(xc,yc,zc));
      printf("df/dz=%f - exact is %f\n",rhs[2]/mat[2][2],f_z(xc,yc,zc));*/

    printf("df/dx=%f - exact is %f\n",rhs[0]/sdm_geom.sum_dx_square,f_x(xc,yc,zc));
    printf("df/dy=%f - exact is %f\n",rhs[1]/sdm_geom.sum_dx_square,f_y(xc,yc,zc));
    printf("df/dz=%f - exact is %f\n",rhs[2]/sdm_geom.sum_dx_square,f_z(xc,yc,zc));

  }

  
} // test_sdm_lsq_gradient

// =====================================================================
// =====================================================================
// =====================================================================
int main(int argc, char* argv[])
{

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI

  int myRank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif // USE_MPI

  
  Kokkos::initialize(argc, argv);

  if (myRank==0) {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
#if defined( CUDA )
    Kokkos::Cuda::print_configuration( msg );
#else
    Kokkos::OpenMP::print_configuration( msg );
#endif
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  if (myRank==0) {
    std::cout << "================================================\n";
    std::cout << "==== Spectral Difference Lagrange      test ====\n";
    std::cout << "==== Least-square gradient estimation ==========\n";
    std::cout << "================================================\n";
  }
  
  
  // testing for multiple values of N between 2 to 6 in 2d and 3d
  {

    // 2d
    test_sdm_lsq_gradient<2,4>();

    // 3d
    test_sdm_lsq_gradient<3,4>();

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}

