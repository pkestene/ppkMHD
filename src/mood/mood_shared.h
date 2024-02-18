#ifndef MOOD_SHARED_H_
#define MOOD_SHARED_H_

#include "shared/kokkos_shared.h"

namespace mood
{

//! data type for the mood pseudo-inverse matrix on DEVICE
using mood_matrix_pi_t = Kokkos::View<real_t **, Device>;

//! data type for the mood pseudo-inverse matrix on HOST
using mood_matrix_pi_host_t = mood_matrix_pi_t::HostMirror;

} // namespace mood

#endif // MOOD_SHARED_H_
