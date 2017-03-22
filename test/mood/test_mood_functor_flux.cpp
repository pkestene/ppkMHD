/**
 * This executable is used to test mood functor for flux computations.
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
#include "mood/mood_utils.h"

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

  // which stencil shall we use ?
  const mood::STENCIL_ID stencilId = mood::STENCIL_2D_DEGREE2;

  // highest degree / order of the polynomial
  unsigned int order = mood::get_stencil_degree(stencilId);
 
  // hydro params
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string("test_mood_functor.ini");
  ConfigMap configMap(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // 2D test
  {

    int dim = 2;
    
    // create fake hydro data
    using DataArray = DataArray2d;
    
    DataArray U        = DataArray("U",1,2,3);
    DataArray Fluxes_x = DataArray("Fx",1,2,3);
    DataArray Fluxes_y = DataArray("Fy",1,2,3);
    DataArray Fluxes_z = DataArray("Fz",0,0,0);

    // instantiate stencil
    mood::Stencil stencil = mood::Stencil(stencilId);

    // create monomial map for all monomial up to degree = order
    mood::MonomialMap monomialMap(dim,order);
    
    // compute the geometric terms matrix and its pseudo-inverse
    int stencil_size   = mood::get_stencil_size(stencilId);
    int stencil_degree = mood::get_stencil_degree(stencilId);
    int NcoefsPolynom = monomialMap.Ncoefs;
    mood::Matrix mat(stencil_size-1,NcoefsPolynom-1);

    std::array<real_t,3> dxyz = {params.dx, params.dy, params.dz};
    printf("dx dy dz : %f %f %f\n",params.dx, params.dy, params.dz);
    mood::fill_geometry_matrix(mat, stencil, monomialMap, dxyz);
    mat.print("geomMatrix");

    // compute geomMatrix pseudo-inverse  and convert it into a Kokkos::View
    mood::Matrix mat_pi;
    mood::compute_pseudo_inverse(mat, mat_pi);
    mat_pi.print("geomMatrix pseudo inverse");

    // create functor
     mood::ComputeFluxesFunctor<2,2,stencilId>
       f(U,Fluxes_x,Fluxes_y,Fluxes_z,params,stencil,mat_pi);
    
    // launch with only 1 thread
    //Kokkos::parallel_for(1,f);

  }

  // 3D test
  // {

  //   using DataArray = DataArray3d;

  //   DataArray U        = DataArray("U",1,2,3,4);
  //   DataArray Fluxes_x = DataArray("Fx",1,2,3,4);
  //   DataArray Fluxes_y = DataArray("Fy",1,2,3,4);
  //   DataArray Fluxes_z = DataArray("Fz",1,2,3,4);
    
  //   // create functor  
  //   TestMoodFunctor<3,order> f(params,U,Fluxes_x,Fluxes_y,Fluxes_z);
    
  //   // launch with only 1 thread
  //   Kokkos::parallel_for(1,f);

  // }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
