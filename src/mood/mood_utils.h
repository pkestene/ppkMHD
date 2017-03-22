#ifndef MOOD_UTILS_H_
#define MOOD_UTILS_H_

// shared
#include "shared/real_type.h"

// mood
#include "mood/Matrix.h"
#include "mood/Stencil.h"
#include "mood/MonomialMap.h"

namespace mood {

/**
 * Fill the Matrix with the geometric terms (for regular cartesian grid).
 *
 * \param[in,out] mat matrix to fill
 * \param[in] stencil
 * \param[in] monomialMap
 */
void fill_geometry_matrix(Matrix& mat,
			  Stencil stencil,
			  const MonomialMap& monomialMap,
			  std::array<real_t,3> dxyz);

} // namespace mood

#endif // MOOD_UTILS_H_
