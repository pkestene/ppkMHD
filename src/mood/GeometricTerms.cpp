#include <math.h>
#include <iostream>

#include "GeometricTerms.h"
#include "Binomial.h"

namespace mood {

// =======================================================
// ==== Class GeometricTerms IMPL ========================
// =======================================================


// =======================================================
// =======================================================
GeometricTerms::GeometricTerms(real_t dx,
			       real_t dy,
			       real_t dz) :
  dx(dx), dy(dy), dz(dz)
{

  // assumes that stencil_radius is in [1,6]
  // so we just need to instanciate the maximun value


  
} // GeometricTerms::GeometricTerms


// =======================================================
// =======================================================
GeometricTerms::~GeometricTerms()
{
  
} // GeometricTerms::~GeometricTerms

// =======================================================
// =======================================================
real_t GeometricTerms::eval_moment(real_t x,
				   real_t y,
				   int n,
				   int m)
{

  return 1.0/dx/dy *
    ( pow(x+dx/0.5,n+1) - pow(x-dx/0.5,n+1) ) / (n+1) *
    ( pow(y+dy/0.5,m+1) - pow(y-dy/0.5,m+1) ) / (m+1);
    
} // GeometricTerms::eval_moment

// =======================================================
// =======================================================
real_t GeometricTerms::eval_moment(real_t x,
				   real_t y,
				   real_t z,
				   int n,
				   int m,
				   int l)
{
  
  return 1.0/dx/dy/dz *
    ( pow(x+dx/0.5,n+1) - pow(x-dx/0.5,n+1) ) / (n+1) *
    ( pow(y+dy/0.5,m+1) - pow(y-dy/0.5,m+1) ) / (m+1) *
    ( pow(z+dz/0.5,l+1) - pow(z-dz/0.5,l+1) ) / (l+1);
    
} // GeometricTerms::eval_moment

// =======================================================
// =======================================================
real_t GeometricTerms::eval_hat(const std::array<real_t,2>& xi,
				const std::array<real_t,2>& xj,
				int n, int m)
{

  real_t res = 0.0;

  for (int b=0; b<m; ++b) {
    for (int a=0; a<n; ++a) {
      res +=
	binom(n,a)*
	binom(m,b)*
	pow(xj[0]-xi[0],a)*
	pow(xj[1]-xi[1],b)*
	eval_moment(xj[0],xj[1],n-a,m-b);
    }
  }

  res -= eval_moment(xi[0],xi[1],n,m);
  
} // GeometricTerms::eval_hat

// =======================================================
// =======================================================
real_t GeometricTerms::eval_hat(const std::array<real_t,3>& xi,
				const std::array<real_t,3>& xj,
				int n, int m, int l)
{

  real_t res = 0.0;

  for (int c=0; c<l; ++c) {
    for (int b=0; b<m; ++b) {
      for (int a=0; a<n; ++a) {
	res +=
	  binom(n,a)*
	  binom(m,b)*
	  binom(l,c)*
	  pow(xj[0]-xi[0],a)*
	  pow(xj[1]-xi[1],b)*
	  pow(xj[2]-xi[2],c)*
	  eval_moment(xj[0],xj[1],xj[2],n-a,m-b,l-c);
      }
    }
  }
  
  res -= eval_moment(xi[0],xi[1],xi[2],n,m,l);

} // GeometricTerms::eval_hat

} // namespace mood

