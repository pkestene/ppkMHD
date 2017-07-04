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
struct EulerEquations
{
  
  //! number of variables: density(1) + energy(1) + momentum(dim)
  static const int nbvar = dim+2;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = Kokkos::Array<real_t,nbvar>;
  
  //! enum
  enum varIDS {
    ID = 0; // density
    IP = 1; // Pressure or Energy
    IU = 2; // momentum along X
    IV = 3; // momentum along Y
    IW = 4; // momentum along Z
  };
  
  //! variables names as a std::map
  std::map<int, std::string>
  get_variable_names()
  {

    std::map<int, std::string> varMap;
    
    varMap[ID] = "rho";
    varMap[IP] = "energy";
    varMap[IU] = "mx"; // momentum component X
    varMap[IV] = "my"; // momentum component Y
    if (dim==3) {
      varMap[IW] = "mz"; // momentum component Z
    }

    return varMap;
    
  } // get_variable_names

  
  
} // namespace ppkMHD

#endif // EULER_EQUATIONS_H_
