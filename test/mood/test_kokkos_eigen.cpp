/**
 * This executable is used to test polynomial reconstruction.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

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

    Kokkos::print_configuration( msg );

    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  using matrix_t = Eigen::MatrixXd;
  using vector_t = Eigen::VectorXd;
  
  matrix_t A = matrix_t::Random(4, 3);
  std::cout << "Here is the matrix A:\n" << A << "\n";
  vector_t b = vector_t::Random(4);
  std::cout << "Here is the right hand side b:\n" << b << "\n";

  // QR decomposition
  Eigen::ColPivHouseholderQR<matrix_t> A_qr = A.colPivHouseholderQr();
  
  // Eigen solve alone 
  {
    
    vector_t sol = A_qr.solve(b);
    std::cout << "[Regular CPU] The QR solve solution is:\n"
    	      << sol << "\n";
    
  }

  // Eigen + Kokkos
  {

    class KokkosEigenTest {
    public:
      KokkosEigenTest(Eigen::ColPivHouseholderQR<matrix_t> A_qr,
		      vector_t rhs,
		      Kokkos::View<double*,DEVICE> sol):
	A_qr(A_qr), rhs(rhs), sol(sol)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int& index) const
      {
	
	vector_t sol_vector = A_qr.solve(rhs);

	for (int i=0; i<sol_vector.size(); ++i) {
	  sol(i) = sol_vector(i);
	}
	
      }
     
      Eigen::ColPivHouseholderQR<matrix_t> A_qr;
      vector_t rhs;
      Kokkos::View<double*,DEVICE> sol;
    };

    Kokkos::View<double*, DEVICE> res("res",3);
    KokkosEigenTest kokkos_eigen_test(A_qr, b,res);
    Kokkos::parallel_for(1, kokkos_eigen_test); 
    
    // retrieve result on Host
    Kokkos::View<double*, DEVICE>::HostMirror res_cpu;
    Kokkos::deep_copy(res_cpu,res);

    std::cout << "[Kokkos] The QR solve solution is:\n"
    	      << res(0) << "\n"
    	      << res(1) << "\n"
    	      << res(2) << "\n";
    
  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
