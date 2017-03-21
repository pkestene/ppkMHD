/**
 * This executable is used to test mood functor.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <type_traits> // for std::conditional

// shared
#include "shared/HydroParams.h" // read parameter file
#include "shared/kokkos_shared.h"
#include "shared/real_type.h"

// mood
#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"
#include "mood/Polynomial.h"
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

#include "mood/MoodFunctors.h"

// dim is the number of variable in the multivariate polynomial representation
constexpr unsigned int dim = 3;

// highest degree / order of the polynomial
constexpr unsigned int order = 4;

// ====================================================
// ====================================================
// ====================================================
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

  // hydro params
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // data array alias
  using DataArray = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  DataArray U        = DataArray("U",1,2,3);
  DataArray Fluxes_x = DataArray("Fx",1,2,3);
  DataArray Fluxes_y = DataArray("Fy",1,2,3);
  DataArray Fluxes_z = DataArray("Fz",1,2,3);
  
  // create functor
  
  mood::ComputeFluxesFunctor<dim,order>(params,U,Fluxes_x,Fluxes_y,Fluxes_z);

  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
