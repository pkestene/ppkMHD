#ifndef MOOD_BASE_QUAD_H_
#define MOOD_BASE_QUAD_H_

#include "shared/kokkos_shared.h"
#include "shared/real_type.h"
#include "shared/enums.h"

namespace mood {

/**
 * A base class which only purpose it to be non-template, so that
 * we can define some static constant data necessary for performing
 * quadrature computations.
 *
 * In the future, if nvcc will support multi-dimensional constexpr array, we
 * will re-use again QuadratureRules.h again.
 *
 * THIS CLASS IS UNUSED, JUST HERE FOR REFERENCE.
 */
class MoodBaseQuad {

public:

  static constexpr real_t SQRT_3   = 1.732050807568877293527446341505872367;
  static constexpr real_t SQRT_3_5 = 0.7745966692414834042779148148838430643;

  static const int maxNbQuadRules  = 3;
  static const int maxNbQuadPoints = 3;
  
  /**
   * array storing the x,y coordinates of the quadrature points
   */
  static const real_t QUAD_LOC_2D[maxNbQuadRules][DIM2][2][maxNbQuadPoints][TWO_D];

  MoodBaseQuad() {};
  virtual ~MoodBaseQuad() {};
  
}; // class MoodBaseQuad

} // namespace mood

#endif // MOOD_BASE_QUAD_H_
