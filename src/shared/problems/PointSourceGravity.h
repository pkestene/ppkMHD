#ifndef POINT_SOURCE_GRAVITY_H_
#define POINT_SOURCE_GRAVITY_H_

#include <math.h>
#include "utils/config/ConfigMap.h"

/**
 * Gravity field due to a point source.
 *
 * Method eval can be used to recompute locally as needed gravity field
 * components.
 */
struct PointSourceGravity {

  //! source location
  real_t xs, ys, zs;

  //! GM is the product of the gravitational contante and the
  //! point source mass.
  real_t GM;

  //! soften parameter
  real_t eps;
  
  PointSourceGravity(ConfigMap& configMap)
  {
    xs = configMap.getFloat("gravity","x",0.0);
    ys = configMap.getFloat("gravity","y",0.0);
    zs = configMap.getFloat("gravity","z",0.0);
    real_t G  = configMap.getFloat("gravity","G",1.0);
    real_t M  = configMap.getFloat("gravity","M",1.0);
    GM = G*M;
    eps = configMap.getFloat("gravity","soften", 0.01);
  }

  /** 
   * Evaluate gravity field components at location P(x,y,z).
   *
   * if \f$ P_s \f$ is point source location, then gravity field at location
   * \f$ P \f$ is given by:
   *
   * \f$ \vec{g} = -\frac{GM}{r^2} \vec{u_r} = \frac{GM}{r^3} \vec{PP_s}\f$
   *
   * \param[in] x,y,z location where gravity is evaluated
   * \param[out] gx,gy,gz gravity field cartesian components
   */
  KOKKOS_INLINE_FUNCTION
  void eval_exact(real_t   x, real_t   y, real_t   z,
		  real_t& gx, real_t& gy, real_t& gz ) const
  {

    real_t r   = sqrt( (x-xs)*(x-xs) + (y-ys)*(y-ys) + (z-zs)*(z-zs) );
    real_t GM2 = GM/(r*r*r);
    
    gx = GM2 * (xs-x);
    gy = GM2 * (ys-y);
    gz = GM2 * (zs-z);
    
  } // eval_exact

  /** 
   * Evaluate gravity field components at location P(x,y,z) with softening
   * to avoid too high gravity field when P is close to source.
   *
   * if \f$ P_s \f$ is point source location, then gravity field at location
   * \f$ P \f$ is given by:
   *
   * \f$ \vec{g} = -\frac{GM}{r^2} \vec{u_r} = \frac{GM}{r^3} \vec{PP_s}\f$
   *
   * where \f$ r \f$ is replace by \f$ \sqrt{r^2+eps^2} \f$
   *
   * \param[in] x,y,z location where gravity is evaluated
   * \param[out] gx,gy,gz gravity field cartesian components
   */
  KOKKOS_INLINE_FUNCTION
  void eval(real_t   x, real_t   y, real_t   z,
	    real_t& gx, real_t& gy, real_t& gz ) const
  {

    real_t r   = sqrt( (x-xs)*(x-xs) + (y-ys)*(y-ys)  + (z-zs)*(z-zs) + eps*eps);
    real_t GM2 = GM/(r*r*r);
    
    gx = GM2 * (xs-x);
    gy = GM2 * (ys-y);
    gz = GM2 * (zs-z);
    
  } // eval
  
}; // struct PointSourceGravity

#endif // POINT_SOURCE_GRAVITY_H_
