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

double test_func(double x, double y)
{
  return x*x+y;
  //return 3.5*x*x+y*0.02+6;
}

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
    using DataArray     = DataArray2d;
    using DataArrayHost = DataArray2dHost;
    
    DataArray U        = DataArray("U",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_x = DataArray("Fx",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_y = DataArray("Fy",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_z = DataArray("Fz",0,0,0);

    DataArrayHost Uh   = Kokkos::create_mirror_view(U);

    printf("Data sizes %d %d\n",params.isize,params.jsize);

    real_t dx = params.dx;
    real_t dy = params.dy;
    
    // init and upload
    printf("U\n");
    for (int i=0; i<params.isize; ++i) {
      for (int j=0; j<params.jsize; ++j) {
	//Uh(i,j,ID) = 1.0*( (i*dx-2*dx)*(i*dx-2*dx) + (j*dy-2*dy)*(j*dy-2*dy) );
	real_t x = dx*(i-2);
	real_t y = dy*(j-2);
	Uh(i,j,ID) = test_func(x,y);
	printf("% 8.5f ",Uh(i,j,ID));
      }
      printf("\n");
    }
    Kokkos::deep_copy(U,Uh);
	
    // instantiate stencil
    mood::Stencil stencil = mood::Stencil(stencilId);

    // create monomial map for all monomial up to degree = order
    mood::MonomialMap monomialMap(dim,order);
    
    // compute the geometric terms matrix and its pseudo-inverse
    int stencil_size   = mood::get_stencil_size(stencilId);
    int stencil_degree = mood::get_stencil_degree(stencilId);
    int NcoefsPolynom = monomialMap.Ncoefs;
    mood::Matrix geomMatrix(stencil_size-1,NcoefsPolynom-1);

    std::array<real_t,3> dxyz = {params.dx, params.dy, params.dz};
    printf("dx dy dz : %f %f %f\n",params.dx, params.dy, params.dz);
    mood::fill_geometry_matrix(geomMatrix, stencil, monomialMap, dxyz);
    //geomMatrix.print("geomMatrix");

    // compute geomMatrix pseudo-inverse  and convert it into a Kokkos::View
    mood::Matrix geomMatrixPI;
    mood::compute_pseudo_inverse(geomMatrix, geomMatrixPI);
    //geomMatrixPI.print("geomMatrix pseudo inverse");

    using geom_t = Kokkos::View<real_t**,DEVICE>;
    using geom_host_t = geom_t::HostMirror;

    geom_t geomMatrixPI_view = geom_t("geomMatrixPI_view",geomMatrixPI.m,geomMatrixPI.n);
    geom_host_t geomMatrixPI_view_h = Kokkos::create_mirror_view(geomMatrixPI_view);

    // copy geomMatrixPI into geomMatrixPI_view
    for (int i = 0; i<geomMatrixPI.m; ++i) { // loop over stencil point
      
      for (int j = 0; j<geomMatrixPI.n; ++j) { // loop over monomial
	
	geomMatrixPI_view_h(i,j) = geomMatrixPI(i,j);
      }
    }
    Kokkos::deep_copy(geomMatrixPI_view, geomMatrixPI_view_h);
    
    // create functor
    mood::ComputeFluxesFunctor<2,2,stencilId>
       f(U,Fluxes_x,Fluxes_y,Fluxes_z,params,stencil, geomMatrixPI_view);
    
    // launch
    int ijsize = params.isize*params.jsize; 
    Kokkos::parallel_for(ijsize,f);

    // retrieve results and print
    printf("Fluxes_x\n");
    Kokkos::deep_copy(Uh,Fluxes_x);
    for (int i=0; i<params.isize; ++i) {
      for (int j=0; j<params.jsize; ++j) {
	real_t x = dx*(i-2) - 0.5*dx;
	real_t y = dy*(j-2);
	printf("% 8.5f ",Uh(i,j,ID) - test_func(x,y));
      }
      printf("\n");
    }
    
  }

  // 3D test
  // {

  //   using DataArray = DataArray3d;

  //   DataArray U        = DataArray("U",params.isize,params.jsize,params.ksize,params.nbvar);
  //   DataArray Fluxes_x = DataArray("Fx",params.isize,params.jsize,params.ksize,params.nbvar);
  //   DataArray Fluxes_y = DataArray("Fy",params.isize,params.jsize,params.ksize,params.nbvar);
  //   DataArray Fluxes_z = DataArray("Fz",params.isize,params.jsize,params.ksize,params.nbvar);
    
  //   // create functor  
  //   TestMoodFunctor<3,order> f(params,U,Fluxes_x,Fluxes_y,Fluxes_z);
    
  //   // launch with only 1 thread
  //   Kokkos::parallel_for(1,f);

  // }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
