/**
 * This executable is used to test class ppkMHD::EulerEquations.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"
#include "shared/EulerEquations.h"

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
void test_2d()
{

  std::cout << "===========\n";
  std::cout << "=====2D====\n";
  std::cout << "===========\n";
  
  
  ppkMHD::EulerEquations<2> eq;

  using HydroState = ppkMHD::EulerEquations<2>::HydroState;

  real_t gamma0 = 1.4;
  
  // conservative variable
  HydroState q;
  q[ID] = 2.3;
  q[IE] = 5;
  q[IU] = 0.4;
  q[IU] = -0.2;

  // primitive variables
  HydroState w;
  eq.convert_to_primitive(q,w,gamma0);

  // speed of sound
  real_t c = eq.compute_speed_of_sound(w, gamma0);

  std::cout << "speed of sound is " << c << "\n";

  // characteristic variables
  HydroState ch;

  // input / out state
  HydroState in, out, out2;
  in[ID] = 0.123;
  in[IU] = 0.456;
  in[IV] = 0.789;
  in[IE] = 0.876;

  eq.cons_to_charac<IX>(in, q, c, gamma0, out);

  eq.charac_to_cons<IX>(out, q, c, gamma0, out2);

  std::cout << "compare out2 to in:\n";
  std::cout << in[ID] << " " << out2[ID] << "\n";
  std::cout << in[IU] << " " << out2[IU] << "\n";
  std::cout << in[IV] << " " << out2[IV] << "\n";
  std::cout << in[IE] << " " << out2[IE] << "\n";
  
  
} // test_2d

void test_3d()
{

  std::cout << "===========\n";
  std::cout << "=====3D====\n";
  std::cout << "===========\n";
  
  ppkMHD::EulerEquations<2> eq;
  
} // test_3d

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


  // instantiate some tests
  test_2d();
  test_3d();

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
