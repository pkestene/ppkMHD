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
template<int dim, int dir>
void test()
{

  std::cout << "===========\n";
  std::cout << "=====" << dim << "D" << "====\n";
  std::cout << "===========\n";
  
  
  ppkMHD::EulerEquations<dim> eq;

  using HydroState = typename ppkMHD::EulerEquations<dim>::HydroState;

  real_t gamma0 = 1.4;
  
  // conservative variable
  HydroState q;
  q[ID] = 2.3;
  q[IE] = 5;
  q[IU] = 0.4;
  q[IV] = -0.2;
  if (dim == 3)
    q[IW] = 1.0;
  
  
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
  in[IE] = 0.876;
  in[IU] = 0.456;
  in[IV] = 0.789;
  if (dim == 3)
    in[IW] = 0.246;

  eq.template cons_to_charac<dir>(in, q, c, gamma0, out);

  eq.template charac_to_cons<dir>(out, q, c, gamma0, out2);

  std::cout << "===============================\n";
  std::cout << "Testing dim=" << dim << " dir=" << dir << "\n";
  std::cout << "compare out2 to in:\n";
  std::cout << in[ID] << " " << out[ID] << " " << out2[ID] << "\n";
  std::cout << in[IE] << " " << out[IE] << " " << out2[IE] << "\n";
  std::cout << in[IU] << " " << out[IU] << " " << out2[IU] << "\n";
  std::cout << in[IV] << " " << out[IV] << " " << out2[IV] << "\n";
  if (dim == 3)
    std::cout << in[IW] << " " << out[IW] << " " << out2[IW] << "\n";
    
} // test

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
  test<2,0>();
  test<2,1>();

  test<3,0>();
  test<3,1>();
  test<3,2>();


  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
