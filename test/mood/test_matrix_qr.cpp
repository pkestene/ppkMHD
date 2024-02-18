/**
 * This executable is used to test Matrix class.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "mood/Matrix.h"

#include "shared/kokkos_shared.h"
#include "shared/real_type.h"

int
main(int argc, char * argv[])
{

  Kokkos::initialize(argc, argv);

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
  }


  // same matrix as in https://rosettacode.org/wiki/QR_decomposition#
  double in[][3] = {
    { 12, -51, 4 }, { 6, 167, -68 }, { -4, 24, -41 }, { -1, 1, 0 }, { 2, 0, 3 },
  };

  mood::Matrix A(in);
  mood::Matrix A_pseudo_inv;

  mood::compute_pseudo_inverse(A, A_pseudo_inv);

  A.print("A");
  A_pseudo_inv.print("A pseudo-inverse");

  // check that pseudo-inv times A = Identity
  mood::Matrix product;
  product.mult(A_pseudo_inv, A);

  product.print("A_pseudo-inv * A (should be Indentity)");

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
