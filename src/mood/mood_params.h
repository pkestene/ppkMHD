#ifndef MOOD_PARAMS_H_
#define MOOD_PARAMS_H_

// mood
#include "mood/Stencil.h"
#include "mood/StencilUtils.h"
#include "mood/GeometricTerms.h"
#include "mood/Matrix.h"

namespace mood
{

/**
 *
 */
struct MoodParams
{

  STENCIL_ID       stencilId;
  Stencil          stencil;
  mood_matrix_pi_t mood_matrix_pi;

  MoodParams(STENCIL_ID stencilId)
    : stencilId(stencilId)
    , Stencil(stencilId)
  {}

}; // MoodParams

} // namespace mood

#endif // MOOD_PARAMS_H_
