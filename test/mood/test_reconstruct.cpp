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
using scalar_t = Kokkos::View<double[1]>;
using scalar_host_t = Kokkos::View<double[1]>::HostMirror;

using coefs_t = Kokkos::Array<real_t,ncoefs>;
using Polynomial_t = mood::Polynomial<dim,order>;

// ====================================================
// ====================================================
// ====================================================
/**
 * A dummy functor to test computation on device with class polynomial.
 */
class TestPolynomialFunctor {

public:

  scalar_t data;
  mood::MonomialMap& monomialMap;
  Kokkos::Array<real_t,dim> eval_point;
  
  TestPolynomialFunctor(scalar_t data,
			mood::MonomialMap& monomialMap,
			Kokkos::Array<real_t,dim> eval_point) :
    data(data),
    monomialMap(monomialMap),
    eval_point(eval_point) {};
  ~TestPolynomialFunctor() {};


  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {

    coefs_t coefs;
    for (int i=0; i<ncoefs; ++i)
      coefs[i] = 1.0*i;

    Polynomial_t polynomial(monomialMap, coefs);

    data(0) = polynomial.eval(eval_point);
    
  }
  
};

// ====================================================
// ====================================================
// ====================================================
/**
 * a simple polynomial test function.
 */
double test_function_2d(double x, double y)
{

  return x*x+2.2*x*y+4.1*y*y-5.0+x;
  
}

// polynomial coefs (for cross-checking)
//coefs_t coefs;


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

  
  
  real_t dx, dy, dz;
  dx = dy = dz = 0.1;
  mood::GeometricTerms geomTerms(dx,dy,dz);

  int stencil_size   = mood::get_stencil_size(stencilId);
  int stencil_degree = mood::get_stencil_degree(stencilId);
  mood::Matrix geomMatrix(stencil_size,stencil_degree);
    
  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
