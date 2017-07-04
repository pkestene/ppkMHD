#ifndef EULER_EQUATIONS_H_
#define EULER_EQUATIONS_H_

#include <map>
#include <array>

#include "shared/real_type.h"

namespace ppkMHD {

/**
 * This structure gather useful information (variable names, 
 * flux functions, ...) for the compressible Euler equations system
 * in both 2D / 3D.
 *
 * Inspired by code dflo (https://github.com/cpraveen/dflo)
 */
template<int dim>
struct EulerEquations {};

/**
 * 2D specialization of the Euler Equation system.
 */
template <>
struct EulerEquations<2>
{
  
  //! number of variables: density(1) + energy(1) + momentum(2)
  static const int nbvar = 2+2;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = Kokkos::Array<real_t,nbvar>;
  
  //! enum
  enum varIDS {
    ID = 0, // density
    IP = 1, // Pressure (when used in primitive variables)
    IE = 1, // Energy
    IU = 2, // momentum along X
    IV = 3, // momentum along Y
    IW = 4, // momentum along Z
  };
  
  //! variables names as a std::map
  static std::map<int, std::string>
  get_variable_names()
  {

    std::map<int, std::string> names;
    
    names[ID] = "rho";
    names[IP] = "energy";
    names[IU] = "mx"; // momentum component X
    names[IV] = "my"; // momentum component Y

    return names;
    
  } // get_variable_names

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction X.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_x(HydroState q, real_t p, HydroState& f)
  {
    f[ID] = q[ID]*q[IU];
    f[IU] = q[ID]*q[IU]*q[IU]+p;
    f[IV] = q[ID]*q[IU]*q[IV];
    f[IE] = q[IU]*(q[IE]+p);
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(HydroState q, real_t p, HydroState& f)
  {
    f[ID] = q[ID]*q[IV];
    f[IU] = q[ID]*q[IV]*q[IU];
    f[IV] = q[ID]*q[IV]*q[IV]+p;
    f[IE] = q[IV]*(q[IE]+p);
  };

  
}; //struct EulerEquations<2>

/**
 * 3D specialization of the Euler Equation system.
 */
template <>
struct EulerEquations<3>
{
  
  //! number of variables: density(1) + energy(1) + momentum(3)
  static const int nbvar = 2+3;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = Kokkos::Array<real_t,nbvar>;
  
  //! enum
  enum varIDS {
    ID = 0, // density
    IP = 1, // Pressure (when used in primitive variables)
    IE = 1, // Energy
    IU = 2, // momentum along X
    IV = 3, // momentum along Y
    IW = 4, // momentum along Z
  };
  
  //! variables names as a std::map
  static std::map<int, std::string>
  get_variable_names()
  {

    std::map<int, std::string> names;
    
    names[ID] = "rho";
    names[IE] = "energy";
    names[IU] = "mx"; // momentum component X
    names[IV] = "my"; // momentum component Y
    names[IW] = "mz"; // momentum component Z

    return names;
    
  } // get_variable_names

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction X.
   *
   * \param[in] vector of conserved variables
   * \param[in] pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_x(HydroState q, real_t p, HydroState& f)
  {
    f[ID] = q[ID]*q[IU];
    f[IU] = q[ID]*q[IU]*q[IU]+p;
    f[IV] = q[ID]*q[IU]*q[IV];
    f[IW] = q[ID]*q[IU]*q[IW];
    f[IE] = q[IU]*(q[IE]+p);
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   *
   * \param[in] vector of conserved variables
   * \param[in] pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(HydroState q, real_t p, HydroState& f)
  {
    f[ID] = q[ID]*q[IV];
    f[IU] = q[ID]*q[IV]*q[IU];
    f[IV] = q[ID]*q[IV]*q[IV]+p;
    f[IW] = q[ID]*q[IV]*q[IW];
    f[IE] = q[IV]*(q[IE]+p);
  };
  
  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Z.
   *
   * \param[in] vector of conserved variables
   * \param[in] pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_z(HydroState q, real_t p, HydroState& f)
  {
    f[ID] = q[ID]*q[IW];
    f[IU] = q[ID]*q[IW]*q[IU];
    f[IV] = q[ID]*q[IW]*q[IV];
    f[IW] = q[ID]*q[IW]*q[IW]+p;
    f[IE] = q[IW]*(q[IE]+p);
  };
  
}; //struct EulerEquations<3>

} // namespace ppkMHD


  
#endif // EULER_EQUATIONS_H_
