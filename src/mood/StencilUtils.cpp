#include "mood/StencilUtils.h"

#include "shared/enums.h"

namespace mood {

// =======================================================
// =======================================================
std::array<std::string,STENCIL_TOTAL_NUMBER> make_stencil_names() {

  std::array<std::string,STENCIL_TOTAL_NUMBER> names;

  names[STENCIL_2D_DEGREE1]    = "STENCIL_2D_DEGREE1";
  names[STENCIL_2D_DEGREE2]    = "STENCIL_2D_DEGREE2";
  names[STENCIL_2D_DEGREE3]    = "STENCIL_2D_DEGREE3";
  names[STENCIL_2D_DEGREE3_V2] = "STENCIL_2D_DEGREE3_V2";
  names[STENCIL_2D_DEGREE4]    = "STENCIL_2D_DEGREE4";
  names[STENCIL_2D_DEGREE5]    = "STENCIL_2D_DEGREE5";
  names[STENCIL_3D_DEGREE1]    = "STENCIL_3D_DEGREE1";
  names[STENCIL_3D_DEGREE2]    = "STENCIL_3D_DEGREE2";
  names[STENCIL_3D_DEGREE3]    = "STENCIL_3D_DEGREE3";
  names[STENCIL_3D_DEGREE4]    = "STENCIL_3D_DEGREE4";
  names[STENCIL_3D_DEGREE5]    = "STENCIL_3D_DEGREE5";
  names[STENCIL_3D_DEGREE5_V2] = "STENCIL_3D_DEGREE5_V2";

  return names;
  
} // make_stencil_names

std::array<std::string,STENCIL_TOTAL_NUMBER> StencilUtils::names = make_stencil_names();

// =======================================================
// =======================================================
std::array<std::string,STENCIL_TOTAL_NUMBER> make_valid_solver_names() {

  std::array<std::string,STENCIL_TOTAL_NUMBER> names;

  names[STENCIL_2D_DEGREE1]    = "Hydro_Mood_2D_degree1";
  names[STENCIL_2D_DEGREE2]    = "Hydro_Mood_2D_degree2";
  names[STENCIL_2D_DEGREE3]    = "Hydro_Mood_2D_degree3";
  names[STENCIL_2D_DEGREE3_V2] = "Hydro_Mood_2D_degree3_V2";
  names[STENCIL_2D_DEGREE4]    = "Hydro_Mood_2D_degree4";
  names[STENCIL_2D_DEGREE5]    = "Hydro_Mood_2D_degree5";
  names[STENCIL_3D_DEGREE1]    = "Hydro_Mood_3D_degree1";
  names[STENCIL_3D_DEGREE2]    = "Hydro_Mood_3D_degree2";
  names[STENCIL_3D_DEGREE3]    = "Hydro_Mood_3D_degree3";
  names[STENCIL_3D_DEGREE4]    = "Hydro_Mood_3D_degree4";
  names[STENCIL_3D_DEGREE5]    = "Hydro_Mood_3D_degree5";
  names[STENCIL_3D_DEGREE5_V2] = "Hydro_Mood_3D_degree5_V2";

  return names;
  
} // make_valid_solver_names

std::array<std::string,STENCIL_TOTAL_NUMBER> StencilUtils::solver_names = make_valid_solver_names();

// =======================================================
// =======================================================
std::string StencilUtils::get_stencil_name(STENCIL_ID id)
{

  return names[id];
  
} // StencilUtils::get_stencil_name

// =======================================================
// =======================================================
STENCIL_ID StencilUtils::get_stencilId_from_string(const std::string& name)
{

  // initialize to unvalid value
  STENCIL_ID result = STENCIL_TOTAL_NUMBER;

  // look into valid names
  for (int i = 0; i<STENCIL_TOTAL_NUMBER; ++i) {
    
    if (!name.compare( names[i] ) ) {
      result = (STENCIL_ID) i;
      break;
    }
    
  }
  return result;
  
} // StencilUtils::get_stencilId_from_string

// =======================================================
// =======================================================
STENCIL_ID StencilUtils::get_stencilId_from_solver_name(const std::string& name)
{

  // initialize to unvalid value
  STENCIL_ID result = STENCIL_TOTAL_NUMBER;

  // look into valid names
  for (int i = 0; i<STENCIL_TOTAL_NUMBER; ++i) {
    
    if (!name.compare( solver_names[i] ) ) {
      result = (STENCIL_ID) i;
      break;
    }
    
  }
  return result;
  
} // StencilUtils::get_stencilId_from_solver_name

// =======================================================
// =======================================================
void StencilUtils::print_stencil(const Stencil& stencil)
{

  std::cout << "############################\n";
  std::cout << "Print stencil " << get_stencil_name(stencil.stencilId) << "\n";
  std::cout << "############################\n";

  for(int i=0; i<stencil.stencilSize; ++i) {

    printf( "[% d, % d,% d]\n",
	    stencil.offsets_h(i,IX),
	    stencil.offsets_h(i,IY),
	    stencil.offsets_h(i,IZ) );
  }
  
} // Stencil::print


} // namespace mood
