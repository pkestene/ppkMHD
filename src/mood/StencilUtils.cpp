#include "mood/StencilUtils.h"

#include "shared/enums.h"

namespace mood {

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
  names[STENCIL_3D_DEGREE3_V2] = "STENCIL_3D_DEGREE3_V2";
  names[STENCIL_3D_DEGREE4]    = "STENCIL_3D_DEGREE4";
  names[STENCIL_3D_DEGREE5]    = "STENCIL_3D_DEGREE5";
  names[STENCIL_3D_DEGREE5_V2] = "STENCIL_3D_DEGREE5_V2";

  return names;
}

std::array<std::string,STENCIL_TOTAL_NUMBER> StencilUtils::names = make_stencil_names();


// =======================================================
// =======================================================
std::string StencilUtils::get_stencil_name(STENCIL_ID id)
{

  return names[id];
  
} // StencilUtils::get_stencil_name

// =======================================================
// =======================================================
void StencilUtils::print_stencil(const Stencil& stencil)
{

  std::cout << "############################\n";
  std::cout << "Print stencil " << get_stencil_name(stencil.stencilId) << "\n";
  std::cout << "############################\n";

  for(int i=0; i<stencil.stencilSize; ++i) {

    std::cout << stencil.offsets_h(i,IX) << " "
	      << stencil.offsets_h(i,IY) << " "
	      << stencil.offsets_h(i,IZ) << "\n"; 
    
  }
  
} // Stencil::print


} // namespace mood
