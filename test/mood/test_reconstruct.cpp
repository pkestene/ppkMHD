/**
 * This executable is used to test polynomial reconstruction.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>


// mood
#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"

#include "mood/Polynomial.h"

// kokkos
#include "shared/real_type.h"

#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

// dim is the number of variable in the multivariate polynomial representation
constexpr int dim = 2;

// highest degree / order of the polynomial
constexpr int order = 2;

// use to initialize data for polynomial
constexpr int ncoefs = mood::binomial<order+dim,order>();

// to be removed ...
using result_t = Kokkos::View<real_t*,DEVICE>;
using result_host_t = result_t::HostMirror;

using geom_t = Kokkos::View<real_t**,DEVICE>;
using geom_host_t = geom_t::HostMirror;

using coefs_t = Kokkos::Array<real_t,ncoefs>;
using Polynomial_t = mood::Polynomial<dim,order>;

using array1d_t = Kokkos::View<real_t*,DEVICE>;

// ====================================================
// ====================================================
// ====================================================
/**
 * a simple polynomial test function.
 */
real_t test_function_2d(real_t x, real_t y)
{

  return x*x+2.2*x*y+4.1*y*y-5.0+x;
  
}

// ====================================================
// ====================================================
// ====================================================
/**
 * A dummy functor to test computation on device with class polynomial
 * and Stencil.
 */
class TestReconstructFunctor {

public:
  
  result_t result;  // size is stencilSize-1
  mood::MonomialMap& monomialMap;
  mood::Stencil stencil;
  mood::STENCIL_ID stencilId;
  geom_t geomPI;
  
  //array1d_t rhs;

  KOKKOS_INLINE_FUNCTION
  real_t test_function_2d_(real_t x, real_t y) const
  {
    
    return x*x+2.2*x*y+4.1*y*y-5.0+x;
    
  }

  TestReconstructFunctor(result_t result,
			 mood::MonomialMap& monomialMap,
			 mood::STENCIL_ID stencilId,
			 geom_t geomPI) :
    result(result),
    monomialMap(monomialMap),
    stencil(stencilId),
    stencilId(stencilId),
    geomPI(geomPI) {

    //int stencil_size   = mood::get_stencil_size(stencilId);

    //rhs = array1d_t("rhs",stencil_size-1); 
  };
  ~TestReconstructFunctor() {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {
    
    Kokkos::Array<real_t,8> rhs;
    for (int ii=0; ii<8; ++ii) {
      rhs[ii]=0;
    }
    
    // assemble RHS
    int s = 0;
    int stencil_size = mood::get_stencil_size(stencilId);
    std::cout << "##### " << stencil_size << "\n";
    
    for (int ii=0; ii<stencil_size; ++ii) {
      int x = stencil.offsets_h(ii,0);
      int y = stencil.offsets_h(ii,1);
      
      if (x != 0 or y != 0) { // avoid stencil center
    	rhs[s] = this->test_function_2d_(x,y) - this->test_function_2d_(0,0);
    	printf("## [% d,% d, % d] % 7.5f\n",x,y,s, rhs[s] + this->test_function_2d_(0,0));
    	s++;
      }
      
    } // end for stencil
    
    // left-multiply by the pseudo-inverse of the geometrical terms matrix.
    // pseudo-inverse matrix has sizes Ncoefs-1, stencil_size-1
    // we will obtain the least-square polynomial solution.
    coefs_t coefs;
    coefs[0] = this->test_function_2d_(0,0);
    for (int ii=0; ii<geomPI.dimension_0(); ++ii) {
      real_t tmp = 0;
      for (int k=0; k<geomPI.dimension_1(); ++k) {
    	tmp += geomPI(ii,k) * rhs[k];
      }
      coefs[ii+1] = tmp;
      printf("device - polynomial [% d] = % 7.5f\n",ii+1,coefs[ii+1]);
    }
    
    Polynomial_t polynomial(monomialMap, coefs);

    for (int ii=0; ii<stencil_size; ++ii) {
      Kokkos::Array<real_t,dim> eval_point;
      eval_point[0] = stencil.offsets_h(ii,0);
      eval_point[1] = stencil.offsets_h(ii,1);

      result(ii) = polynomial.eval(eval_point);
      //std::cout << polynomial.eval(eval_point) << "\n";
    }
    
  }
  
}; // class TestReconstructFunctor


// void set_coefs_2d(const mood::MonomialMap& MonomialMap, int e1, int e2, real_t value)
// {  
// }
  
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

  // create monomial map for all monomial up to degree = order
  mood::MonomialMap monomialMap(dim,order);


  /*
   * Select a stencil.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Stencil       \n";
  std::cout << "############################\n";

  mood::STENCIL_ID stencilId = mood::Stencil::select_stencil(dim,order);

  mood::Stencil stencil = mood::Stencil(stencilId);

  mood::StencilUtils::print_stencil(stencil);

  // compute on host
  coefs_t coefs;
  for (int i=0; i<ncoefs; ++i)
    coefs[i] = 0.0;

  Polynomial_t polynomial(monomialMap, coefs);

  // set coefs so that it is the same as in
  // x*x + 2.2*x*y + 4.1*y*y -5.0 + x;
  Kokkos::Array<int,3> e; 
  e = {2,0,0};
  polynomial.setCoefs(e,1.0);
  e = {1,1,0};
  polynomial.setCoefs(e,2.2);
  e = {0,2,0};
  polynomial.setCoefs(e,4.1);
  e = {0,0,0};
  polynomial.setCoefs(e,-5.0);
  e = {1,0,0};
  polynomial.setCoefs(e,1.0);

  std::cout << "Print polynomial coefs\n";
  for (int ii=0; ii<monomialMap.Ncoefs-1; ++ii)
    printf( "polynomial [% d] = % 7.5f\n",ii,coefs[ii+1]);
  printf("\n");
  
  real_t dx, dy, dz;
  dx = dy = dz = 1.0;
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int stencil_size   = mood::get_stencil_size(stencilId);
  int stencil_degree = mood::get_stencil_degree(stencilId);
  int NcoefsPolynom = monomialMap.Ncoefs;
  mood::Matrix geomMatrix(stencil_size-1,NcoefsPolynom-1);
  
  // fill geomMatrix
  int i=0;
  for (int ii = 0; ii<stencil_size; ++ii) { // loop over stencil point
    int x = stencil.offsets_h(ii,0);
    int y = stencil.offsets_h(ii,1);

    if (x != 0 or y != 0) { // avoid stencil center
   
      printf("stencil point coordiantes : % d,% d\n",x,y);
      
      for (int j = 0; j<geomMatrix.n; ++j) { // loop over monomial
	// stencil point
	// get monomial exponent for j+1 (to avoid the constant term)
	int n = monomialMap.data_h(j+1,0);
	int m = monomialMap.data_h(j+1,1);
	
	geomMatrix(i,j) = geomTerms.eval_hat(x,y,n,m);
      }
      
      ++i;
    }
  }
  geomMatrix.print("geomMatrix");

  // compute geomMatrix pseudo-inverse  and convert it into a Kokkos::View
  mood::Matrix geomMatrixPI;
  mood::compute_pseudo_inverse(geomMatrix, geomMatrixPI);
  geomMatrixPI.print("geomMatrix pseudo inverse");

  // check that pseudo-inv times A = Identity
  mood::Matrix product;
  product.mult(geomMatrixPI, geomMatrix);
  product.print("geomMatrixPI * geomMatrix (should be Indentity)");

  printf("\n");
  std::array<real_t,8> rhs_host;
  int rhs_index=0;
  std::cout << "test function computed on host\n";
  {
    // print test function value on host
    for (int index=0; index < stencil_size; ++index) {
      int x = stencil.offsets_h(index,0);
      int y = stencil.offsets_h(index,1);

      if (x != 0 or y !=0) {
	printf("[% d,% d]  % 7.5f\n",x,y,test_function_2d(x,y) - test_function_2d(0,0));

	rhs_host[rhs_index] = test_function_2d(x,y) - test_function_2d(0,0);
	rhs_index++;
      }
    }
  }
  
  // instantiate functor and compute interpolating polynomial
  result_t result = result_t("result",stencil_size);
  result_host_t result_h = Kokkos::create_mirror_view(result);

  // try retrieve polynomial coefficient on host: multiply geomMatrixPI
  // by rhs_host
  Kokkos::Array<real_t,6> coef_host;
  coef_host[0] = test_function_2d(0,0);
  
  for (int index=0; index<geomMatrixPI.m; index++) {
    real_t tmp=0;
    for (int k=0; k<geomMatrixPI.n; k++) {
      tmp += geomMatrixPI(index,k) * rhs_host[k]; 
    }
    coef_host[index+1] = tmp;
  }

  std::cout << "polynomial coef obtained by least-square fit\n";
  for (int ii=0; ii<monomialMap.Ncoefs; ++ii)
    std::cout << "polynomial [" << ii << "] = " << coef_host[ii] << "\n";

  Polynomial_t polynomial_host(monomialMap, coef_host);
  // cross-check polynomial
  if (dim == 2) {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      printf("% 7.5f * {% d,% d}   // X^% d * Y^% d\n",
	     coef_host[i],e[0],e[1],e[0],e[1]);
      
    }
  } else {
    for (int i = 0; i<monomialMap.Ncoefs; ++i) {
      
      int e[3] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1),
		  monomialMap.data_h(i,2)};
      printf("% 7.5f * {% d,% d,% d}   // X^% d * Y^% d * Z^% d\n",
	     coef_host[i],e[0],e[1],e[2],e[0],e[1],e[2]);
      
    }
  }

  
  for (int ii=0; ii<stencil_size; ++ii) {
    Kokkos::Array<real_t,dim> eval_point;
    eval_point[0] = stencil.offsets_h(ii,0);
    eval_point[1] = stencil.offsets_h(ii,1);

    int x = stencil.offsets_h(ii,0);
    int y = stencil.offsets_h(ii,1);
    printf("polynomial eval on host [% d, % d]  % 7.5f == % 7.5f\n",x,y,
	   polynomial_host.eval(eval_point),
	   test_function_2d(x, y) );
  }

  /*
   *
   */
  geom_t geomMatrixPI_view = geom_t("geomMatrixPI_view",geomMatrixPI.m,geomMatrixPI.n);
  geom_host_t geomMatrixPI_view_h = Kokkos::create_mirror_view(geomMatrixPI_view);

  // copy geomMatrixPI into geomMatrixPI_view
  for (int ii = 0; ii<geomMatrixPI.m; ++ii) { // loop over stencil point
    
    for (int j = 0; j<geomMatrixPI.n; ++j) { // loop over monomial
      
      geomMatrixPI_view_h(ii,j) = geomMatrixPI(ii,j);
    }
  }
  Kokkos::deep_copy(geomMatrixPI_view, geomMatrixPI_view_h);
  
  // create functor
  TestReconstructFunctor f(result, monomialMap, stencilId, geomMatrixPI_view);

  // launch with only 1 thread
  Kokkos::parallel_for(1,f);

  // get back results
  Kokkos::deep_copy(result_h, result);

  // print results
  std::cout << "test function computed on device\n";
  {
    // print test function value on host
    for (int index=0; index < stencil_size; ++index) {
      int x = stencil.offsets_h(index,0);
      int y = stencil.offsets_h(index,1);

      if (x != 0 or y != 0) { // avoid stencil center
	printf("[% d,% d]  % 7.5f\n",x,y,result_h(index));
      }
    }
  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
