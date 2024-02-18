#include "mood/MoodBaseQuad.h"

namespace mood
{


// initialize static members
const real_t MoodBaseQuad::QUAD_LOC_2D[maxNbQuadRules][DIM2][2][maxNbQuadPoints][TWO_D] = {
  // 1 quadrature point (items #2 and #3 are not used)
  {   // along X
    { // -X
      { { -0.5, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } },
      // +X
      { { 0.5, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } } },

    // along Y
    { // -Y
      { { 0.0, -0.5 }, { 0.0, 0.0 }, { 0.0, 0.0 } },
      // +Y
      { { 0.0, 0.5 }, { 0.0, 0.0 }, { 0.0, 0.0 } } } }, // end 1 quadrature point

  // 2 quadrature points (item #3 is not used)
  {   // along X
    { // -X
      { { -0.5, -0.5 / SQRT_3 }, { -0.5, 0.5 / SQRT_3 }, { 0.0, 0.0 } },
      // +X
      { { 0.5, -0.5 / SQRT_3 }, { 0.5, 0.5 / SQRT_3 }, { 0.0, 0.0 } } },
    // along Y
    { // -Y
      { { -0.5 / SQRT_3, -0.5 }, { 0.5 / SQRT_3, -0.5 }, { 0.0, 0.0 } },
      // +Y
      { { -0.5 / SQRT_3, 0.5 },
        { 0.5 / SQRT_3, 0.5 },
        { 0.0, 0.0 } } } }, // end 2 quadrature points

  // 3 quadrature points
  {   // along X
    { // -X
      {
        { -0.5, -0.5 * SQRT_3_5 },
        { -0.5, 0.0 },
        { -0.5, 0.5 * SQRT_3_5 },
      },

      // +X
      {
        { 0.5, -0.5 * SQRT_3_5 },
        { 0.5, 0.0 },
        { 0.5, 0.5 * SQRT_3_5 },
      } },

    // along Y
    { // -Y
      {
        { -0.5 * SQRT_3_5, -0.5 },
        { 0.0, -0.5 },
        { 0.5 * SQRT_3_5, -0.5 },
      },

      // +Y
      {
        { -0.5 * SQRT_3_5, 0.5 },
        { 0.0, 0.5 },
        { 0.5 * SQRT_3_5, 0.5 },
      } }

  } // end 3 quadrature points

}; // end QUAD_LOC_2D

} // namespace mood
