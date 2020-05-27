/**
 * This executable is used to test polynomial reconstruction.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>

// kokkos
#include "shared/real_type.h"

// mood
#include "mood/monomials_ordering.h"
#include "mood/monomials_print_utils.h"
#include "mood/Polynomial.h"


#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

// dim is the number of variable in the multivariate polynomial representation
constexpr unsigned int dim = 2;

// highest degree of the polynomial
constexpr unsigned int degree = 5;

// use to initialize data for polynomial
constexpr int ncoefs = mood::binomial<degree+dim,degree>();

constexpr mood::STENCIL_ID stencilId_ = mood::STENCIL_2D_DEGREE5;

// to be removed ...
using result_t = Kokkos::View<real_t*,Device>;
using result_host_t = result_t::HostMirror;

using geom_t = Kokkos::View<real_t**,Device>;
using geom_host_t = geom_t::HostMirror;

using coefs_t = Kokkos::Array<real_t,ncoefs>;

using array1d_t = Kokkos::View<real_t*,Device>;

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

template<int N>
real_t polynomial_eval(real_t x, real_t y,
		       Kokkos::View<int**,Device>::HostMirror monomMap,
		       Kokkos::Array<real_t,N> coefs)  {
  
  real_t result = 0;
  
  // span monomial degrees
  for (int i = 0; i<N; ++i) {
    int e[2] = {monomMap(i,0),
		monomMap(i,1)};
    result += coefs[i] * pow(x,e[0]) * pow(y,e[1]);
  }
  
  return result;
  
}; // eval 2D

template<int N>
real_t polynomial_eval(real_t x, real_t y, real_t z,
		       Kokkos::View<int**,Device>::HostMirror monomMap,
		       Kokkos::Array<real_t,N> coefs)  {
  
  real_t result = 0;
  
  // span monomial degrees
  for (int i = 0; i<N; ++i) {
    int e[3] = {monomMap(i,0),
		monomMap(i,1),
		monomMap(i,2)};
    result += coefs[i] * pow(x,e[0]) * pow(y,e[1]) * pow(z,e[2]);
  }
  
  return result;
  
}; // eval 3D

// ====================================================
// ====================================================
// ====================================================
/**
 * A dummy functor to test computation on device with class polynomial
 * and Stencil.
 */
template <mood::STENCIL_ID stencilId>
class TestReconstructFunctor : public mood::PolynomialEvaluator<dim,degree> {

public:
  
  result_t result;  // size is stencilSize-1
  mood::Stencil stencil;
  geom_t geomPI;
  
  //array1d_t rhs;

  KOKKOS_INLINE_FUNCTION
  real_t test_function_2d_(real_t x, real_t y) const
  {
    
    return x*x+2.2*x*y+4.1*y*y-5.0+x;
    
  }

  TestReconstructFunctor(result_t result,
			 geom_t geomPI,
			 typename mood::MonomialMap<dim,degree>::MonomMap monomMap) :
    PolynomialEvaluator<dim,degree>(monomMap),
    result(result),
    stencil(stencilId),
    geomPI(geomPI) {};
  ~TestReconstructFunctor() {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {
    
    Kokkos::Array<real_t,stencil_size-1> rhs;
    for (int ii=0; ii<rhs.size(); ++ii) {
      rhs[ii]=0;
    }
    
    // assemble RHS
    int s = 0;
#ifndef CUDA
    std::cout << "[DEVICE] stencil_size is " << stencil_size << "\n";
#endif
    
    for (int ii=0; ii<stencil_size; ++ii) {
      int x = stencil.offsets(ii,0);
      int y = stencil.offsets(ii,1);
      
      if (x != 0 or y != 0) { // avoid stencil center
    	rhs[s] = this->test_function_2d_(x,y) - this->test_function_2d_(0,0);
#ifndef CUDA
    	printf("[DEVICE] [% d,% d,% d] % 7.5f\n",x,y,s, rhs[s] + this->test_function_2d_(0,0));
#endif
    	s++;
      }
      
    } // end for stencil
    
    // left-multiply by the pseudo-inverse of the geometrical terms matrix.
    // pseudo-inverse matrix has sizes Ncoefs-1, stencil_size-1
    // we will obtain the least-square polynomial solution.
    coefs_t coefs;
    coefs[0] = this->test_function_2d_(0,0);
    for (int ii=0; ii<geomPI.extent(0); ++ii) {
      real_t tmp = 0;
      for (int ik=0; ik<geomPI.extent(1); ++ik) {
    	tmp += geomPI(ii,ik) * rhs[ik];
      }
      coefs[ii+1] = tmp;
#ifndef CUDA
      printf("[DEVICE] - polynomial [% d] = % 7.5f\n",ii+1,coefs[ii+1]);
#endif
    }
    
    for (int ii=0; ii<stencil_size; ++ii) {
      Kokkos::Array<real_t,dim> eval_point;
      eval_point[0] = stencil.offsets(ii,0);
      eval_point[1] = stencil.offsets(ii,1);

      result(ii) = eval(eval_point, coefs);
#ifndef CUDA
      // std::cout << "[DEVICE] results - "
      // 		<< polynomial_eval<ncoefs>(eval_point[0],
      // 					   eval_point[1],
      // 					   monomMap.data, coefs) << "\n";
#endif
    }
    
  } // operator()

  // get the number of cells in stencil
  static constexpr int stencil_size = mood::STENCIL_SIZE[stencilId];

}; // class TestReconstructFunctor

  
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
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

  // create monomial map for all monomial up to degree = degree
  mood::MonomialMap<dim,degree> monomialMap;


  /*
   * Select a stencil.
   */
  std::cout << "############################\n";
  std::cout << "Testing class Stencil       \n";
  std::cout << "############################\n";

  mood::Stencil stencil = mood::Stencil(stencilId_);

  mood::StencilUtils::print_stencil(stencil);

  // compute on host
  coefs_t coefs;
  for (int i=0; i<ncoefs; ++i)
    coefs[i] = 0.0;

  // set coefs so that it is the same as in
  // x*x + 2.2*x*y + 4.1*y*y -5.0 + x;
  Kokkos::Array<int,3> e;
  e = {2,0,0};
  mood::polynomial_setCoefs<ncoefs,degree>(coefs, monomialMap, e[0], e[1],  1.0);
  e = {1,1,0};
  mood::polynomial_setCoefs<ncoefs,degree>(coefs, monomialMap, e[0], e[1],  2.2);
  e = {0,2,0};
  mood::polynomial_setCoefs<ncoefs,degree>(coefs, monomialMap, e[0], e[1],  4.1);
  e = {0,0,0};
  mood::polynomial_setCoefs<ncoefs,degree>(coefs, monomialMap, e[0], e[1], -5.0);
  e = {1,0,0};
  mood::polynomial_setCoefs<ncoefs,degree>(coefs, monomialMap, e[0], e[1],  1.0);

  std::cout << "Print polynomial coefs\n";
  for (int ii=0; ii<mood::MonomialMap<dim,degree>::ncoefs-1; ++ii)
    printf( "polynomial [% d] = % 7.5f\n",ii,coefs[ii+1]);
  printf("\n");
  
  real_t dx, dy, dz;
  dx = dy = dz = 1.0;
  mood::GeometricTerms geomTerms(dx,dy,dz);

  constexpr int stencil_size = mood::STENCIL_SIZE[stencilId_];
  int stencil_degree = mood::get_stencil_degree(stencilId_);
  int NcoefsPolynom = mood::MonomialMap<dim,degree>::ncoefs;
  mood::Matrix geomMatrix(stencil_size-1,NcoefsPolynom-1);
  
  // fill geomMatrix
  int i=0;
  for (int ii = 0; ii<stencil_size; ++ii) { // loop over stencil point
    int x = stencil.offsets_h(ii,0);
    int y = stencil.offsets_h(ii,1);

    if (x != 0 or y != 0) { // avoid stencil center
   
      printf("stencil point coordinates : % d,% d\n",x,y);
      
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
  std::array<real_t,stencil_size-1> rhs_host;
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
  Kokkos::Array<real_t,ncoefs> coef_host;
  coef_host[0] = test_function_2d(0,0);
  
  for (int index=0; index<geomMatrixPI.m; index++) {
    real_t tmp=0;
    for (int ik=0; ik<geomMatrixPI.n; ik++) {
      tmp += geomMatrixPI(index,ik) * rhs_host[ik]; 
    }
    coef_host[index+1] = tmp;
  }

  std::cout << "polynomial coef obtained by least-square fit\n";
  for (int ii=0; ii<mood::MonomialMap<dim,degree>::ncoefs; ++ii)
    std::cout << "polynomial [" << ii << "] = " << coef_host[ii]
	      << " (diff = " << coef_host[ii] - coefs[ii] << ")" << "\n";

  // cross-check polynomial
  if (dim == 2) {
    for (int i = 0; i<mood::MonomialMap<dim,degree>::ncoefs; ++i) {
      
      int e[2] = {monomialMap.data_h(i,0),
		  monomialMap.data_h(i,1)};
      printf("% 7.5f * {% d,% d}   // X^% d * Y^% d\n",
	     coef_host[i],e[0],e[1],e[0],e[1]);
      
    }
  } else {
    for (int i = 0; i<mood::MonomialMap<dim,degree>::ncoefs; ++i) {
      
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
	   polynomial_eval<ncoefs>(eval_point[0], eval_point[1],
				   monomialMap.data_h, coefs),
	   test_function_2d(x, y));
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
  
  // create monomial map for all monomial up to degree = degree
  mood::MonomialMap<dim,degree> monomialMap2;

  // create functor
  TestReconstructFunctor<stencilId_> f(result,
				       geomMatrixPI_view,
				       monomialMap2.data);

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
	printf("[% d,% d]  % 7.5f == % 7.5f\n",x,y,result_h(index),test_function_2d(x,y));
      }
    }
  }
  
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
