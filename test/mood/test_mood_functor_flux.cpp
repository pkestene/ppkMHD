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

#include "mood/MoodFluxesFunctors.h"
#include "mood/mood_utils.h"

double test_func(double x, double y)
{
  return x*x+y;
  //return 3.5*x*x+y*0.02+6;
}

double test_func_3d(double x, double y, double z)
{
  return x*x+y-2*z*z;
}

namespace mood {

/**
 * Compute MOOD fluxes.
 * 
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template<int dim,
	 int degree,
	 STENCIL_ID stencilId>
class TestFunctor : public MoodBaseFunctor<dim,degree>
{
    
public:
  using typename MoodBaseFunctor<dim,degree>::DataArray;
  using typename PolynomialEvaluator<dim,degree>::coefs_t;
  
  /**
   * Constructor for 2D/3D.
   */
  TestFunctor(DataArray        Udata,
	      DataArray        FluxData_x,
	      DataArray        FluxData_y,
	      DataArray        FluxData_z,
	      HydroParams      params,
	      Stencil          stencil,
	      mood_matrix_pi_t mat_pi) :
    MoodBaseFunctor<dim,degree>(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z),
    stencil(stencil),
    mat_pi(mat_pi)
  {};

  ~TestFunctor() {};

  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);
    
    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;

    // rhs for neighbor cell (accross an x-face, y-face or z-face)
    //Kokkos::Array<real_t,stencil_size-1> rhs_n;
    
    
    if(j >= ghostWidth && j < jsize-ghostWidth+1  &&
       i >= ghostWidth && i < isize-ghostWidth+1 ) {

      // retrieve neighbors data for ID, and build rhs
      int irhs = 0;
      for (int is=0; is<stencil_size; ++is) {
	int x = stencil.offsets(is,0);
	int y = stencil.offsets(is,1);
	if (x != 0 or y != 0) {
	  rhs[irhs] = Udata(i+x,j+y,ID) - Udata(i,j,ID);
	  irhs++;
	}	
      } // end for is

      // retrieve reconstruction polynomial coefficients in current cell
      coefs_t coefs_c;
      coefs_c[0] = Udata(i,j,ID);
      for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	real_t tmp = 0;
	for (int k=0; k<mat_pi.dimension_1(); ++k) {
	  tmp += mat_pi(icoef,k) * rhs[k];
	}
	coefs_c[icoef+1] = tmp;
      }

      // reconstruct Udata on the left face along X direction
      // for each quadrature points
      //if (nbQuadraturePoints==1) {
      //int x = QUADRATURE_LOCATION_2D_N1_X_M[0][IX];
      //int y = QUADRATURE_LOCATION_2D_N1_X_M[0][IY];
      //}

      FluxData_x(i,j,ID) = this->eval(-0.5*dx, 0.0   ,coefs_c);
      FluxData_y(i,j,ID) = this->eval( 0.0   ,-0.5*dy,coefs_c);
      
    } // end if
    
  } // end functor 2d

  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;
    
    if(k >= ghostWidth && k < ksize - ghostWidth+1 &&
       j >= ghostWidth && j < jsize - ghostWidth+1 &&
       i >= ghostWidth && i < isize - ghostWidth+1) {

      // retrieve neighbors data for ID, and build rhs
      int irhs = 0;
      for (int is=0; is<stencil_size; ++is) {
	int x = stencil.offsets(is,0);
	int y = stencil.offsets(is,1);
	int z = stencil.offsets(is,2);
	if (x != 0 or y != 0 or z != 0) {
	  rhs[irhs] = Udata(i+x,j+y,k+z,ID) - Udata(i,j,k,ID);
	  irhs++;
	}	
      } // end for is

      // retrieve reconstruction polynomial coefficients in current cell
      coefs_t coefs_c;
      coefs_c[0] = Udata(i,j,k,ID);
      for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	real_t tmp = 0;
	for (int ik=0; ik<mat_pi.dimension_1(); ++ik) {
	  tmp += mat_pi(icoef,ik) * rhs[ik];
	}
	coefs_c[icoef+1] = tmp;
      }

      // reconstruct Udata on the left face along X direction
      // for each quadrature points
      //if (nbQuadraturePoints==1) {
      //int x = QUADRATURE_LOCATION_3D_N1_X_M[0][IX];
      //int y = QUADRATURE_LOCATION_3D_N1_X_M[0][IY];
      //int z = QUADRATURE_LOCATION_3D_N1_X_M[0][IZ];
      //}

      FluxData_x(i,j,k,ID) = this->eval(-0.5*dx, 0.0   , 0.0   , coefs_c);
      FluxData_y(i,j,k,ID) = this->eval( 0.0   ,-0.5*dy, 0.0   , coefs_c);
      FluxData_z(i,j,k,ID) = this->eval( 0.0   , 0.0   ,-0.5*dz, coefs_c);

      
    }
    
  }  // end functor 3d
  
  DataArray        Udata;
  DataArray        FluxData_x, FluxData_y, FluxData_z;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;

  // get the number of cells in stencil
  static constexpr int stencil_size = STENCIL_SIZE[stencilId];

  // get the number of quadrature point per face corresponding to this stencil
  static constexpr int nbQuadraturePoints = QUADRATURE_NUM_POINTS[stencilId];
  
}; // class TestFunctor

} // namespace mood


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
 
  // 2D test
  {
    
    constexpr int dim = 2;

    // which stencil shall we use ?
    constexpr mood::STENCIL_ID stencilId = mood::STENCIL_2D_DEGREE5;
    
    // degree of the polynomial
    constexpr int degree = mood::STENCIL_DEGREE[stencilId];
    
    // hydro params
    // read parameter file and initialize parameter
    // parse parameters from input file
    std::string input_file = std::string("test_mood_functor_2d.ini");
    ConfigMap configMap(input_file);
    
    // test: create a HydroParams object
    HydroParams params = HydroParams();
    params.setup(configMap);
    
    // make sure ghostWidth is ok
    params.ghostWidth = mood::get_stencil_ghostwidth(stencilId);
    params.isize = params.nx + 2*params.ghostWidth;
    params.jsize = params.ny + 2*params.ghostWidth;
    params.print();
    std::string solver_name = configMap.getString("run", "solver_name", "unknown");
    std::cout << "Using solver mood with degree" << degree << "\n";
    
    // create fake hydro data
    using DataArray     = DataArray2d;
    using DataArrayHost = DataArray2dHost;
    
    DataArray U        = DataArray("U",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_x = DataArray("Fx",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_y = DataArray("Fy",params.isize,params.jsize,params.nbvar);
    DataArray Fluxes_z = DataArray("Fz",0,0,0);

    DataArrayHost Uh   = Kokkos::create_mirror_view(U);

    printf("Data sizes %d %d - ghostwidth=%d\n",params.isize,params.jsize,params.ghostWidth);

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

    // create monomial map for all monomial up to degree = degree
    mood::MonomialMap<dim,degree> monomialMap;
    
    // compute the geometric terms matrix and its pseudo-inverse
    int stencil_size   = mood::get_stencil_size(stencilId);
    int stencil_degree = mood::get_stencil_degree(stencilId);
    int NcoefsPolynom = mood::MonomialMap<dim,degree>::ncoefs;
    mood::Matrix geomMatrix(stencil_size-1,NcoefsPolynom-1);

    std::array<real_t,3> dxyz = {params.dx, params.dy, params.dz};
    printf("dx dy dz : %f %f %f\n",params.dx, params.dy, params.dz);
    mood::fill_geometry_matrix<dim,degree>(geomMatrix, stencil, monomialMap, dxyz);
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
    mood::TestFunctor<dim,degree,stencilId>
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

  std::cout << "// ======================== //\n";
  std::cout << "// ======================== //\n";
  
  // 3D test
  {
    
    constexpr int dim = 3;
    
    // which stencil shall we use ?
    constexpr mood::STENCIL_ID stencilId = mood::STENCIL_3D_DEGREE2;
    
    // degree of the polynomial
    constexpr int degree = mood::STENCIL_DEGREE[stencilId];

    // hydro params
    // read parameter file and initialize parameter
    // parse parameters from input file
    std::string input_file = std::string("test_mood_functor_3d.ini");
    ConfigMap configMap(input_file);
    
    // test: create a HydroParams object
    HydroParams params = HydroParams();
    params.setup(configMap);
    
    // make sure ghostWidth is ok
    params.ghostWidth = mood::get_stencil_ghostwidth(stencilId);
    params.isize = params.nx + 2*params.ghostWidth;
    params.jsize = params.ny + 2*params.ghostWidth;
    params.print();
    std::string solver_name = configMap.getString("run", "solver_name", "unknown");
    std::cout << "Using solver " << solver_name << "\n";
    
    // create fake hydro data
    using DataArray     = DataArray3d;
    using DataArrayHost = DataArray3dHost;
    
    DataArray U        = DataArray("U",params.isize,params.jsize,params.ksize,params.nbvar);
    DataArray Fluxes_x = DataArray("Fx",params.isize,params.jsize,params.ksize,params.nbvar);
    DataArray Fluxes_y = DataArray("Fy",params.isize,params.jsize,params.ksize,params.nbvar);
    DataArray Fluxes_z = DataArray("Fz",params.isize,params.jsize,params.ksize,params.nbvar);

    DataArrayHost Uh   = Kokkos::create_mirror_view(U);
    
    printf("Data sizes %d %d %d - ghostwidth=%d\n",
	   params.isize,
	   params.jsize,
	   params.ksize,
	   params.ghostWidth);

    real_t dx = params.dx;
    real_t dy = params.dy;
    real_t dz = params.dz;
    
    // init and upload
    printf("U\n");
    for (int i=0; i<params.isize; ++i) {
      for (int j=0; j<params.jsize; ++j) {
	for (int k=0; k<params.ksize; ++k) {
	  //Uh(i,j,ID) = 1.0*( (i*dx-2*dx)*(i*dx-2*dx) + (j*dy-2*dy)*(j*dy-2*dy) );
	  real_t x = dx*(i-2);
	  real_t y = dy*(j-2);
	  real_t z = dz*(k-2);
	  Uh(i,j,k,ID) = test_func_3d(x,y,z);
	  //printf("% 8.5f ",Uh(i,j,k,ID));
	}
	//printf("\n");
      }
    }
    Kokkos::deep_copy(U,Uh);
	
    // instantiate stencil
    mood::Stencil stencil = mood::Stencil(stencilId);

    // create monomial map for all monomial up to degree = degree
    mood::MonomialMap<dim,degree> monomialMap;
    
    // compute the geometric terms matrix and its pseudo-inverse
    int stencil_size   = mood::get_stencil_size(stencilId);
    int stencil_degree = mood::get_stencil_degree(stencilId);
    int NcoefsPolynom = mood::MonomialMap<dim,degree>::ncoefs;
    mood::Matrix geomMatrix(stencil_size-1,NcoefsPolynom-1);
    
    std::array<real_t,3> dxyz = {params.dx, params.dy, params.dz};
    printf("dx dy dz : %f %f %f\n",params.dx, params.dy, params.dz);
    mood::fill_geometry_matrix<dim,degree>(geomMatrix, stencil, monomialMap, dxyz);
    geomMatrix.print("geomMatrix");

    // compute geomMatrix pseudo-inverse  and convert it into a Kokkos::View
    mood::Matrix geomMatrixPI;
    mood::compute_pseudo_inverse(geomMatrix, geomMatrixPI);
    geomMatrixPI.print("geomMatrix pseudo inverse");

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
    
    // create functor - template params are dim,degree,stencil
    mood::TestFunctor<dim,degree,stencilId>
       f(U,Fluxes_x,Fluxes_y,Fluxes_z,params,stencil, geomMatrixPI_view);
    
    // launch
    int ijksize = params.isize*params.jsize*params.ksize; 
    Kokkos::parallel_for(ijksize,f);

    // retrieve results and print
    printf("Fluxes_x\n");
    Kokkos::deep_copy(Uh,Fluxes_x);
    for (int i=0; i<params.isize; ++i) {
      for (int j=0; j<params.jsize; ++j) {
	for (int k=0; k<params.ksize; ++k) {
	  real_t x = dx*(i-2) - 0.5*dx;
	  real_t y = dy*(j-2);
	  real_t z = dz*(k-2);
	  printf("% 8.5f ",Uh(i,j,k,ID) - test_func_3d(x,y,z));
	}
	printf("\n");
      }
      printf("---\n");
    }
    
  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
