#ifndef WAVE_PARAMS_H_
#define WAVE_PARAMS_H_

#include "utils/config/ConfigMap.h"

struct WaveParams {

  // wave problem parameters
  real_t wave_amplitude;
  real_t wave_V0;
  /* see https://www.astro.princeton.edu/~jstone/Athena/tests/linear-waves/mhd-right-eigenvector.html
   * 0 : fast magnetosonic wave
   * 1 : Alfven wave
   * 2 : slow magnetosonic wave
   * 3 : Contact wave
   */
  int wave_type;
  
  /* right eigenvector */
  real_t rev[7];

  real_t d0, p0;
  real_t sin_a2, cos_a2, sin_a3, cos_a3;
  real_t kpar;
  real_t dby,dbz;
  real_t bx0, by0, bz0;
  real_t k_par;

  WaveParams(ConfigMap& configMap)
  {
    double xmin = configMap.getFloat("mesh", "xmin", 0.0);
    double ymin = configMap.getFloat("mesh", "ymin", 0.0);
    double zmin = configMap.getFloat("mesh", "zmin", 0.0);

    double xmax = configMap.getFloat("mesh", "xmax", 3.0);
    double ymax = configMap.getFloat("mesh", "ymax", 1.5);
    double zmax = configMap.getFloat("mesh", "zmax", 1.5);
    
    double Lx = xmax - xmin;
    double Ly = ymax - ymin;
    double Lz = zmax - zmin;
    
    double gamma0 = configMap.getFloat("hydro", "gamma0", 1.66667);
    
    wave_amplitude   = configMap.getFloat("wave","amplitude", 1.0e-6);
    wave_type   = configMap.getInteger("wave","type", 0);

    /* rev are the eigenvectors for the individual wave.
     * Calculated using the output of Athena's linear wave problem:
     * https://github.com/PrincetonUniversity/Athena-Cversion/blob/master/tst/3D-mhd/athinput.linear_wave3d
     */

    /* fast wave */
    if (wave_type == 0) {
      rev[0] = 4.472136e-01;
      rev[1] = -8.944272e-01;
      rev[2] = 4.216370e-01;
      rev[3] = 1.490712e-01;
      rev[4] = 2.012461e+00;
      rev[5] = 8.432740e-01;
      rev[6] = 2.981424e-01;
      wave_V0   = 0.0;
    /* Alfven wave */
    } else if (wave_type == 1) {
      rev[0] = 0.0;
      rev[1] = 0.0;
      rev[2] = -3.333333e-01;
      rev[3] = 9.428090e-01;
      rev[4] = 0.0;
      rev[5] = -3.333333e-01;
      rev[6] = 9.428090e-01;
      wave_V0   = 0.0;
    /* Slow wave */
    } else if (wave_type == 2) {
      rev[0] = 8.944272e-01; 
      rev[1] = -4.472136e-01;
      rev[2] = -8.432740e-01;
      rev[3] = -2.981424e-01;
      rev[4] = 6.708204e-01;
      rev[5] = -4.216370e-01;
      rev[6] = -1.490712e-01;
      wave_V0   = 0.0;
    /* Contact wave */
    } else if (wave_type == 3) {
      rev[0] = 1.0;
      rev[1] = 1.0;
      rev[2] = 0.0;
      rev[3] = 0.0;
      rev[4] = 0.5;
      rev[5] = 0.0;
      rev[6] = 0.0;
      wave_V0   = 1.0;
    } else {
      std::cerr << "wave_type = " << wave_type << " not implemented!\nABORT!\n";
      exit(EXIT_FAILURE);
    }

    /* Initialization adapted from
     * https://github.com/PrincetonUniversity/athena-public-version/blob/master/src/pgen/linear_wave.cpp
     * Changeset 77ea410
     * authored by James M. Stone and other code contributors under 3-clause BSD License
     * */
    d0 = 1.0;
    p0 = 1.0/gamma0;
    
    const double TwoPi = 4.0*asin(1.0);
    
    real_t ang_3 = atan(Lx/Ly);
    sin_a3 = sin(ang_3);
    cos_a3 = cos(ang_3);

    real_t ang_2 = atan(0.5*(Lx*cos_a3 + Ly*sin_a3)/Lz);
    sin_a2 = sin(ang_2);
    cos_a2 = cos(ang_2);
    
    real_t x1 = Lx*cos_a2*cos_a3;
    real_t x2 = Ly*cos_a2*sin_a3;
    real_t x3 = Lz*sin_a2;
    
    real_t lambda = x1;
    if (ang_3 != 0.0)
      lambda = fmin(lambda,x2);
    if (ang_2 != 0.)
      lambda = fmin(lambda,x3);
    
    // k_parallel
    k_par = TwoPi/lambda;

    dby = wave_amplitude * rev[5];
    dbz = wave_amplitude * rev[6];

    bx0 = 1.0;
    by0 = sqrt(2.0);
    bz0 = 0.5;
    
  }

}; // struct WaveParams

#endif // WAVE_PARAMS_H_
