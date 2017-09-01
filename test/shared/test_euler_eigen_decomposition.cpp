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

  // conservative in/ characterics / conservative out states
  HydroState cons_in, charac, cons_out;
  cons_in[ID] = 0.123;
  cons_in[IE] = 0.876;
  cons_in[IU] = 0.456;
  cons_in[IV] = 0.789;
  if (dim == 3)
    cons_in[IW] = 0.246;

  // copy cons_in into cons_out
  cons_out[ID] = cons_in[ID];
  cons_out[IE] = cons_in[IE];
  cons_out[IU] = cons_in[IU];
  cons_out[IV] = cons_in[IV];
  if (dim == 3)
    cons_out[IW] = cons_in[IW];

  
  // computation done in place
  eq.template cons_to_charac<dir>(cons_out, q, c, gamma0);

  // copy to characteristics variables
  charac[ID] = cons_out[ID];
  charac[IE] = cons_out[IE];
  charac[IU] = cons_out[IU];
  charac[IV] = cons_out[IV];
  if (dim == 3)
    charac[IW] = cons_out[IW];
  
  eq.template charac_to_cons<dir>(cons_out, q, c, gamma0);

  std::cout << "===============================\n";
  std::cout << "Testing dim=" << dim << " dir=" << dir << "\n";
  std::cout << "compare cons_out to cons_in:\n";
  std::cout << cons_in[ID] << " " << charac[ID] << " " << cons_out[ID] << " " << fabs(cons_in[ID]-cons_out[ID]) << "\n";
  std::cout << cons_in[IE] << " " << charac[IE] << " " << cons_out[IE] << " " << fabs(cons_in[IE]-cons_out[IE]) << "\n";
  std::cout << cons_in[IU] << " " << charac[IU] << " " << cons_out[IU] << " " << fabs(cons_in[IU]-cons_out[IU]) << "\n";
  std::cout << cons_in[IV] << " " << charac[IV] << " " << cons_out[IV] << " " << fabs(cons_in[IV]-cons_out[IV]) << "\n";
  if (dim == 3)
    std::cout << cons_in[IW] << " " << charac[IW] << " " << cons_out[IW] << " " << fabs(cons_in[IW]-cons_out[IW]) << "\n";
    
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
