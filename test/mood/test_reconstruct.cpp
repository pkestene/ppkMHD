/**
 * This executable is used to test polynomila reconstruction.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"

// mood
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"

// eigen3
#include <Eigen/Dense>

int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);

  {
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

  // dim is the number of variable in the multivariate polynomial representation
  unsigned int dim=3;

  // highest degree / order of the polynomial
  int order = 4;
  
  if (argc>1)
    dim = atoi(argv[1]);
  if (argc>2)
    order = atoi(argv[2]);

  mood::STENCIL_ID stencilId = mood::Stencil::select_stencil(dim,order);

  /*
   * test class Polynomial.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Stencil    \n";
  std::cout << "############################\n";

  mood::Stencil stencil = mood::Stencil(stencilId);

  //mood::StencilUtils::print_stencil(stencil);

  real_t dx, dy, dz;
  dx = dy = dz = 0.1;
  mood::GeometricTerms geomTerms(dx,dy,dz);


  // Eigen linear solver example
  {

    using matrix_t = Eigen::MatrixXd;
    using vector_t = Eigen::VectorXd;
    
    matrix_t A = matrix_t::Random(4, 3);
    std::cout << "Here is the matrix A:\n" << A << "\n";
    vector_t b = vector_t::Random(4);
    std::cout << "Here is the right hand side b:\n" << b << "\n";

    // JacobiSVD
    vector_t sol = A.jacobiSvd(Eigen::ComputeThinU |
    			       Eigen::ComputeThinV).solve(b);
    std::cout << "The least-squares (SVD) solution is:\n"
    	      << sol << "\n";

    // QR solve
    // print QR
    Eigen::ColPivHouseholderQR<matrix_t> A_qr = A.colPivHouseholderQr();

    matrix_t pseudo_inv = A_qr.inverse();
    
    // std::cout << "Pseudo inverse is\n";
    // std::cout << pseudo_inv << "\n";
    
    // vector_t sol2 = pseudo_inv*b;
    // std::cout << "The QR pseudo-inv x b is:\n"
    // 	      << sol2 << "\n";
    
    vector_t sol3 = A.colPivHouseholderQr().solve(b);
    std::cout << "The QR solve solution is:\n"
    	      << sol3 << "\n";

    // diff
    std::cout << "The difference between the 2 solutions is:\n"
    	      << sol3 - sol << "\n";
    
    
    
  }
    
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
