/**
 * This is a minimal working example of eigen.
 *
 * g++ --std=c++11 -O3 -I/usr/include/eigen3 test_eigen.cpp -o test_eigen
 *
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

// eigen3
#include <Eigen/Dense>

int
main(int argc, char * argv[])
{
  // Eigen linear solver example
  using matrix_t = Eigen::MatrixXd;
  using vector_t = Eigen::VectorXd;

  matrix_t A = matrix_t::Random(4, 3);
  std::cout << "Here is the matrix A:\n" << A << "\n";
  vector_t b = vector_t::Random(4);
  std::cout << "Here is the right hand side b:\n" << b << "\n";

  // JacobiSVD
  vector_t sol = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  std::cout << "The least-squares (SVD) solution is:\n" << sol << "\n";

  // QR solve
  // print QR
  Eigen::ColPivHouseholderQR<matrix_t> A_qr = A.colPivHouseholderQr();

  vector_t sol2 = A_qr.solve(b);
  std::cout << "The QR solve solution is:\n" << sol2 << "\n";

  // diff
  std::cout << "The difference between the 2 solutions is:\n" << sol2 - sol << "\n";


  // info
  std::cout << std::boolalpha;
  std::cout << "Is QR decomposition class POD ? "
            << std::is_pod<Eigen::ColPivHouseholderQR<matrix_t>>::value << '\n';

  return EXIT_SUCCESS;
}
