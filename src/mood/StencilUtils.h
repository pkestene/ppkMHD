#ifndef MOOD_STENCIL_UTILS_H_
#define MOOD_STENCIL_UTILS_H_

#include "mood/Stencil.h"

#include <array>
#include <string>

namespace mood
{

/**
 *
 */
struct StencilUtils
{

  StencilUtils(){};

  static std::string
  get_stencil_name(STENCIL_ID id);
  static STENCIL_ID
  get_stencilId_from_string(const std::string & name);

  static std::array<std::string, STENCIL_TOTAL_NUMBER> names;
  static std::array<std::string, STENCIL_TOTAL_NUMBER> solver_names;

  static void
  print_stencil(const Stencil & stencil);
};

inline STENCIL_ID
get_stencilId_from_solver_name(const std::string &                                   name,
                               const std::array<std::string, STENCIL_TOTAL_NUMBER> & solver_names)
{
  // initialize to unvalid value
  STENCIL_ID result = STENCIL_TOTAL_NUMBER;

  // look into valid names
  for (int i = 0; i < STENCIL_TOTAL_NUMBER; ++i)
  {

    if (!name.compare(solver_names[i]))
    {
      result = (STENCIL_ID)i;
      break;
    }
  }
  return result;

} // get_stencilId_from_solver_name


// =======================================================
// =======================================================
inline std::array<std::string, STENCIL_TOTAL_NUMBER>
make_stencil_names()
{

  std::array<std::string, STENCIL_TOTAL_NUMBER> names;

  names[STENCIL_2D_DEGREE1] = "STENCIL_2D_DEGREE1";
  names[STENCIL_2D_DEGREE2] = "STENCIL_2D_DEGREE2";
  names[STENCIL_2D_DEGREE3] = "STENCIL_2D_DEGREE3";
  names[STENCIL_2D_DEGREE3_V2] = "STENCIL_2D_DEGREE3_V2";
  names[STENCIL_2D_DEGREE4] = "STENCIL_2D_DEGREE4";
  names[STENCIL_2D_DEGREE5] = "STENCIL_2D_DEGREE5";
  names[STENCIL_3D_DEGREE1] = "STENCIL_3D_DEGREE1";
  names[STENCIL_3D_DEGREE2] = "STENCIL_3D_DEGREE2";
  names[STENCIL_3D_DEGREE3] = "STENCIL_3D_DEGREE3";
  names[STENCIL_3D_DEGREE4] = "STENCIL_3D_DEGREE4";
  names[STENCIL_3D_DEGREE5] = "STENCIL_3D_DEGREE5";
  names[STENCIL_3D_DEGREE5_V2] = "STENCIL_3D_DEGREE5_V2";

  return names;

} // make_stencil_names

// =======================================================
// =======================================================
inline std::array<std::string, STENCIL_TOTAL_NUMBER>
make_valid_solver_names()
{

  std::array<std::string, STENCIL_TOTAL_NUMBER> names;

  names[STENCIL_2D_DEGREE1] = "Hydro_Mood_2D_degree1";
  names[STENCIL_2D_DEGREE2] = "Hydro_Mood_2D_degree2";
  names[STENCIL_2D_DEGREE3] = "Hydro_Mood_2D_degree3";
  names[STENCIL_2D_DEGREE3_V2] = "Hydro_Mood_2D_degree3_V2";
  names[STENCIL_2D_DEGREE4] = "Hydro_Mood_2D_degree4";
  names[STENCIL_2D_DEGREE5] = "Hydro_Mood_2D_degree5";
  names[STENCIL_3D_DEGREE1] = "Hydro_Mood_3D_degree1";
  names[STENCIL_3D_DEGREE2] = "Hydro_Mood_3D_degree2";
  names[STENCIL_3D_DEGREE3] = "Hydro_Mood_3D_degree3";
  names[STENCIL_3D_DEGREE4] = "Hydro_Mood_3D_degree4";
  names[STENCIL_3D_DEGREE5] = "Hydro_Mood_3D_degree5";
  names[STENCIL_3D_DEGREE5_V2] = "Hydro_Mood_3D_degree5_V2";

  return names;

} // make_valid_solver_names


} // namespace mood

#endif // MOOD_STENCIL_UTILS_H_
