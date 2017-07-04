#ifndef EULER_EQUATIONS_H_
#define EULER_EQUATIONS_H_

#include <map>
#include <array>

#include "shared/real_type.h"
#include "shared/enums.h"

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

  //! numeric constant 1/2
  static constexpr real_t ONE_HALF = 0.5;
  
  //! number of variables: density(1) + energy(1) + momentum(2)
  static const int nbvar = 2+2;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = Kokkos::Array<real_t,nbvar>;
  
  //! enum
  // static const int ID = 0; // density
  // static const int IP = 1; // Pressure (when used in primitive variables)
  // static const int IE = 1; // Energy
  // static const int IU = 2; // momentum along X
  // static const int IV = 3; // momentum along Y

  //! velocity gradient tensor number of components
  static const int nbvar_grad = 2*2;
  
  //! velocity gradient components
  enum gradientV_IDS {
    U_X = 0,
    U_Y = 1,
    V_X = 2,
    V_Y = 3
  };

  //! alias typename to an array holding gradient velocity tensor components
  using GradTensor = Kokkos::Array<real_t,nbvar_grad>;

  //! just a dim-dimension vector
  using Vector = Kokkos::Array<real_t,2>;
  
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
  void flux_x(HydroState q, real_t p, HydroState& flux)
  {
    flux[ID] = q[ID]*q[IU];
    flux[IU] = q[ID]*q[IU]*q[IU]+p;
    flux[IV] = q[ID]*q[IU]*q[IV];
    flux[IE] = q[IU]*(q[IE]+p);
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(HydroState q, real_t p, HydroState& flux)
  {
    flux[ID] = q[ID]*q[IV];
    flux[IU] = q[ID]*q[IV]*q[IU];
    flux[IV] = q[ID]*q[IV]*q[IV]+p;
    flux[IE] = q[IV]*(q[IE]+p);
  };

  /**
   * Viscous term as a flux along direction X.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion 
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_v_x(GradTensor g, Vector v, Vector f, real_t mu,
		HydroState& flux)
  {
    real_t tau_xx = 2*mu*(g[U_X]-ONE_HALF*(g[U_X]+g[V_Y]));
    real_t tau_xy =   mu*(g[V_X] + g[U_Y]);
    
    flux[ID] = 0.0;
    flux[IU] = tau_xx;
    flux[IV] = tau_xy;
    flux[IE] = v[IX]*tau_xx + v[IY]*tau_xy + f[IX];
  };
  
  /**
   * Viscous term as a flux along direction Y.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion 
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_v_y(GradTensor g, Vector v, Vector f, real_t mu,
		HydroState& flux)
  {
    real_t tau_yy = 2*mu*(g[V_Y]-ONE_HALF*(g[U_X]+g[V_Y]));
    real_t tau_xy =   mu*(g[V_X] + g[U_Y]);
    
    flux[ID] = 0.0;
    flux[IU] = tau_xy;
    flux[IV] = tau_yy;
    flux[IE] = v[IX]*tau_xy + v[IY]*tau_yy + f[IY];
  };
  
}; //struct EulerEquations<2>

/**
 * 3D specialization of the Euler Equation system.
 */
template <>
struct EulerEquations<3>
{
  
  //! numeric constant 1/3
  static constexpr real_t ONE_THIRD = 1.0/3;

  //! number of variables: density(1) + energy(1) + momentum(3)
  static const int nbvar = 2+3;

  //! type alias to a small array holding hydrodynamics state variables
  using HydroState = Kokkos::Array<real_t,nbvar>;
  
  //! enum
  // enum varIDS {
  //   ID = 0, // density
  //   IP = 1, // Pressure (when used in primitive variables)
  //   IE = 1, // Energy
  //   IU = 2, // momentum along X
  //   IV = 3, // momentum along Y
  //   IW = 4, // momentum along Z
  // };
  
  //! velocity gradient tensor number of components
  static const int nbvar_grad = 3*3;

  //! velocity gradient components
  enum gradientV_IDS {
    U_X = 0,
    U_Y = 1,
    U_Z = 2,
    V_X = 3,
    V_Y = 4,
    V_Z = 5,
    W_X = 6,
    W_Y = 7,
    W_Z = 8
  };

  //! alias typename to an array holding gradient velocity tensor components
  using GradTensor = Kokkos::Array<real_t,nbvar_grad>;

  //! just a dim-dimension vector
  using Vector = Kokkos::Array<real_t,3>;

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
   * \param[in] q vector of conserved variables
   * \param[in] p pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_x(HydroState q, real_t p, HydroState& flux)
  {
    flux[ID] = q[ID]*q[IU];
    flux[IU] = q[ID]*q[IU]*q[IU]+p;
    flux[IV] = q[ID]*q[IU]*q[IV];
    flux[IW] = q[ID]*q[IU]*q[IW];
    flux[IE] = q[IU]*(q[IE]+p);
  };

  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Y.
   *
   * \param[in] q vector of conserved variables
   * \param[in] p pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_y(HydroState q, real_t p, HydroState& flux)
  {
    flux[ID] = q[ID]*q[IV];
    flux[IU] = q[ID]*q[IV]*q[IU];
    flux[IV] = q[ID]*q[IV]*q[IV]+p;
    flux[IW] = q[ID]*q[IV]*q[IW];
    flux[IE] = q[IV]*(q[IE]+p);
  };
  
  /**
   * Flux expression in the Euler equations system written in conservative
   * form along direction Z.
   *
   * \param[in] q vector of conserved variables
   * \param[in] p pressure (of computed by the fluid equation of state)
   * \param[out] flux vector
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_z(HydroState q, real_t p, HydroState& flux)
  {
    flux[ID] = q[ID]*q[IW];
    flux[IU] = q[ID]*q[IW]*q[IU];
    flux[IV] = q[ID]*q[IW]*q[IV];
    flux[IW] = q[ID]*q[IW]*q[IW]+p;
    flux[IE] = q[IW]*(q[IE]+p);
  };
  
  /**
   * Viscous term as a flux along direction X.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion 
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_v_x(GradTensor g, Vector v, Vector f, real_t mu,
		HydroState& flux)
  {
    real_t tau_xx = 2*mu*(g[U_X]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));
    real_t tau_yx =   mu*(g[V_X] + g[U_Y]);
    real_t tau_zx =   mu*(g[W_X] + g[U_Z]);
    
    flux[ID] = 0.0;
    flux[IU] = tau_xx;
    flux[IV] = tau_yx;
    flux[IW] = tau_zx;
    flux[IE] = v[IX]*tau_xx + v[IY]*tau_yx + v[IZ]*tau_zx + f[IX];
  };

  /**
   * Viscous term as a flux along direction Y.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion 
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_v_y(GradTensor g, Vector v, Vector f, real_t mu,
		HydroState& flux)
  {
    real_t tau_xy =   mu*(g[U_Y] + g[V_X]);
    real_t tau_yy = 2*mu*(g[V_Y]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));
    real_t tau_zy =   mu*(g[W_Y] + g[V_Z]);
    
    flux[ID] = 0.0;
    flux[IU] = tau_xy;
    flux[IV] = tau_yy;
    flux[IW] = tau_zy;
    flux[IE] = v[IX]*tau_xy + v[IY]*tau_yy + v[IZ]*tau_zy + f[IY];
  };

  /**
   * Viscous term as a flux along direction Z.
   *
   * \param[in] g is the velocity gradient tensor
   * \param[in] v is the hydrodynamics velocity vector
   * \param[in] f is a vector (gradient of diffusive term)
   * \param[in] mu is dynamics viscosity (mu = rho * nu)
   * \param[out]
   *
   * note that the diffusive term f represents thermal + entropy diffusion 
   * as in ASH / CHORUS code.
   */
  static
  KOKKOS_INLINE_FUNCTION
  void flux_v_z(GradTensor g, Vector v, Vector f, real_t mu,
		HydroState& flux)
  {
    real_t tau_xz =   mu*(g[U_Z] + g[W_X]);
    real_t tau_yz =   mu*(g[V_Z] + g[W_Y]);
    real_t tau_zz = 2*mu*(g[W_Z]-ONE_THIRD*(g[U_X]+g[V_Y]+g[W_Z]));
    
    flux[ID] = 0.0;
    flux[IU] = tau_xz;
    flux[IV] = tau_yz;
    flux[IW] = tau_zz;
    flux[IE] = v[IX]*tau_xz + v[IY]*tau_yz + v[IZ]*tau_zz + f[IZ];
  };

}; //struct EulerEquations<3>

} // namespace ppkMHD


  
#endif // EULER_EQUATIONS_H_
