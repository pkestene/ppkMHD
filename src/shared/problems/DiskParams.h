#ifndef DISK_PARAMS_H_
#define DISK_PARAMS_H_

#include "utils/config/ConfigMap.h"

namespace ppkMHD
{

/**
 * disk problem parameters.
 */
struct DiskParams
{

  //! disk radius
  real_t radius;

  //! disk inner radius (where eos is isothermal)
  real_t radius_inner;

  //! disk center location
  real_t xc;
  real_t yc;
  real_t zc;

  //! reference density
  real_t ref_density;

  //! contrast density (in the outer region versus peripheral region)
  real_t contrast_density;

  //! radial distance (after the end of the disk) where density drops by
  //! a factor of contrast_density
  real_t contrast_width;

  real_t ref_sound_speed;

  DiskParams(ConfigMap & configMap)
  {

    double xmin = configMap.getFloat("mesh", "xmin", -0.5);
    double ymin = configMap.getFloat("mesh", "ymin", -0.5);
    double zmin = configMap.getFloat("mesh", "zmin", -0.5);

    double xmax = configMap.getFloat("mesh", "xmax", 0.5);
    double ymax = configMap.getFloat("mesh", "ymax", 0.5);
    double zmax = configMap.getFloat("mesh", "zmax", 0.5);

    radius = configMap.getFloat("disk", "radius", (xmax - xmin) * 0.125);
    radius_inner = configMap.getFloat("disk", "radius_inner", radius / 10);
    xc = configMap.getFloat("disk", "xc", (xmin + xmax) / 2);
    yc = configMap.getFloat("disk", "yc", (ymin + ymax) / 2);
    zc = configMap.getFloat("disk", "zc", (zmin + zmax) / 2);

    ref_density = configMap.getFloat("disk", "ref_density", 1.0);
    contrast_density = configMap.getFloat("disk", "contrast_density", 100.0);
    contrast_width = configMap.getFloat("disk", "contrast_width", 0.01);
    ref_sound_speed = configMap.getFloat("disk", "ref_sound_speed", 0.2);
  }

  KOKKOS_INLINE_FUNCTION
  real_t
  radial_speed_of_sound(real_t r) const
  {

    real_t csound = 0;

    // saturate at the center to prevent diverging values
    if (r < radius_inner)
    {
      csound = ref_sound_speed / sqrt(radius_inner / radius);
    }
    else
    {
      csound = ref_sound_speed / sqrt(r / radius);
    }

    return csound;

  } // radial_speed_of_sound

}; // struct DiskParams

} // namespace ppkMHD

#endif // DISK_PARAMS_H_
