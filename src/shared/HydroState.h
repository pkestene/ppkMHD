#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"

constexpr int HYDRO_2D_NBVAR=4;
constexpr int HYDRO_3D_NBVAR=5;
constexpr int MHD_2D_NBVAR=8;
constexpr int MHD_3D_NBVAR=8;
constexpr int MHD_NBVAR=8;

using HydroState2d = Kokkos::Array<real_t,HYDRO_2D_NBVAR>;
using HydroState3d = Kokkos::Array<real_t,HYDRO_3D_NBVAR>;
using MHDState     = Kokkos::Array<real_t,MHD_NBVAR>;
using BField       = Kokkos::Array<real_t,3>;

//!< a POD data structure to store local conservative / primitive variables (hydro 2d)
// struct HydroState2d {
//   real_t d;
//   real_t p;
//   real_t u;
//   real_t v;

//   KOKKOS_INLINE_FUNCTION
//   HydroState2d() : d(0.0), p(0.0), u(0.0), v(0.0) {}
// };

//!< a POD data structure to store local conservative / primitive variables (hydro 3d)
// struct HydroState3d {
//   real_t d;
//   real_t p;
//   real_t u;
//   real_t v;
//   real_t w;

//   KOKKOS_INLINE_FUNCTION
//   HydroState3d() : d(0.0), p(0.0), u(0.0), v(0.0), w(0.0) {}
// };

//!< a POD data structure to store local conservative / primitive variables (mhd 2d and 3d)
// struct MHDState {
//   real_t d;
//   real_t p;
//   real_t u;
//   real_t v;
//   real_t w;
//   real_t bx;
//   real_t by;
//   real_t bz;

//   KOKKOS_INLINE_FUNCTION
//   MHDState() : d(0.0), p(0.0), u(0.0), v(0.0), w(0.0), bx(0.0), by(0.0), bz(0.0) {}
// };

//!< a POD data structure to store magnetif field
// struct BField {

//   real_t bx;
//   real_t by;
//   real_t bz;

//   KOKKOS_INLINE_FUNCTION
//   BField() : bx(0.0), by(0.0), bz(0.0) {}
// };

#endif // HYDRO_STATE_H_
