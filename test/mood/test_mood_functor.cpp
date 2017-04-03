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

#include "mood/MoodBaseFunctor.h"

// highest degree of the polynomial
constexpr unsigned int degree_ = 4;


/**
 *
 * Test mood functor.
 */
template<unsigned int dim, unsigned int degree>
class TestMoodFunctor : public mood::MoodBaseFunctor<dim,degree>
{
    
public:
  using typename mood::MoodBaseFunctor<dim,degree>::DataArray;

  
  /**
   * Constructor for 2D/3D.
   */
  TestMoodFunctor(HydroParams params,
		  DataArray Udata,
		  DataArray FluxData_x,
		  DataArray FluxData_y,
		  DataArray FluxData_z) :
    mood::MoodBaseFunctor<dim,degree>(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z)    
  {};

  ~TestMoodFunctor() {};

  //! functor for 2d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& i)  const
  {
#ifndef CUDA
    printf("2D functor\n");
#endif
  }

  //! functor for 3d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& i) const
  {
#ifndef CUDA
    printf("3D functor\n");
#endif
  }
  
  DataArray Udata;
  DataArray FluxData_x, FluxData_y, FluxData_z;

}; // class TestMoodFunctor


// ====================================================
// ====================================================
// ====================================================
int main(int argc, char* argv[])
{

  if (argc < 2)
    Kokkos::abort("[ABORT] You MUST provide a parameter file !");

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

  // 2D test
  {

    // data array alias
    //using DataArray = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
    using DataArray = DataArray2d;
    
    DataArray U        = DataArray("U",1,2,3);
    DataArray Fluxes_x = DataArray("Fx",1,2,3);
    DataArray Fluxes_y = DataArray("Fy",1,2,3);
    DataArray Fluxes_z = DataArray("Fz",0,0,0);

    // create functor  
    TestMoodFunctor<2,degree_> f(params,U,Fluxes_x,Fluxes_y,Fluxes_z);
    
    // launch with only 1 thread
    Kokkos::parallel_for(1,f);

  }

  // 3D test
  {

    using DataArray = DataArray3d;

    DataArray U        = DataArray("U",1,2,3,4);
    DataArray Fluxes_x = DataArray("Fx",1,2,3,4);
    DataArray Fluxes_y = DataArray("Fy",1,2,3,4);
    DataArray Fluxes_z = DataArray("Fz",1,2,3,4);
    
    // create functor  
    TestMoodFunctor<3,degree_> f(params,U,Fluxes_x,Fluxes_y,Fluxes_z);
    
    // launch with only 1 thread
    Kokkos::parallel_for(1,f);

  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
