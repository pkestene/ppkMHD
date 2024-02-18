#ifndef ORSZAG_TANG_PARAMS_H_
#define ORSZAG_TANG_PARAMS_H_

#include <math.h>

#include "utils/config/ConfigMap.h"

namespace ppkMHD
{

struct OrszagTangParams
{

  // transverse wave vector
  real_t kt;

  OrszagTangParams(ConfigMap & configMap) { kt = configMap.getFloat("OrszagTang", "kt", 0.0); }

}; // struct OrszagTangParams

} // namespace ppkMHD

#endif // ORSZAG_TANG_PARAMS_H_
