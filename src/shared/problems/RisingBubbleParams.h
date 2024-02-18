#ifndef RISING_BUBBLE_PARAMS_H_
#define RISING_BUBBLE_PARAMS_H_

#include "utils/config/ConfigMap.h"
#include "shared/real_type.h"

namespace ppkMHD
{

/**
 * Rising bubble test parameters.
 */
struct RisingBubbleParams
{

  // bubble parameters
  real_t R;
  real_t x0, y0, z0;
  real_t din, dout;  // density inside / outside
  real_t gx, gy, gz; // uniform initial gravity field


  RisingBubbleParams(ConfigMap & configMap)
  {

    R = configMap.getFloat("rising_bubble", "R", 0.1);
    x0 = configMap.getFloat("rising_bubble", "x0", 0.5);
    y0 = configMap.getFloat("rising_bubble", "y0", 0.25);
    z0 = configMap.getFloat("rising_bubble", "z0", 0.5);
    din = configMap.getFloat("rising_bubble", "din", 1.0);
    dout = configMap.getFloat("rising_bubble", "dout", 2.0);

    gx = configMap.getFloat("rising_bubble", "gx", 0.0);
    gy = configMap.getFloat("rising_bubble", "gy", -0.1);
    gz = configMap.getFloat("rising_bubble", "gz", 0.0);
  }

}; // struct RisingBubbleParams

} // namespace ppkMHD

#endif // RISING_BUBBLE_PARAMS_H_
