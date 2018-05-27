#ifndef GRESHO_VORTEX_PARAMS_H_
#define GRESHO_VORTEX_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * The Gresho problem is a rotating vortex problem independent of time
 * for the case of inviscid flow (Euler equations).
 *
 * reference : https://www.cfd-online.com/Wiki/Gresho_vortex
 */
struct GreshoParams {

  real_t rho0;
  real_t Ma;

  GreshoParams(ConfigMap& configMap)
  {

    rho0  = configMap.getFloat("Gresho", "rho0", 1.0);
    Ma    = configMap.getFloat("Gresho", "Ma",   0.1);

  } // GreshoParams
  
}; // struct GreshoParams

#endif // GRESHO_VORTEX_PARAMS_H_
