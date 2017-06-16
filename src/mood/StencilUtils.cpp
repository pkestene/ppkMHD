#include "mood/StencilUtils.h"

#include "shared/enums.h"

namespace mood {

// =======================================================
// =======================================================
// std::array<std::string,STENCIL_TOTAL_NUMBER> make_stencil_names() {

  
// } // make_stencil_names

std::array<std::string,STENCIL_TOTAL_NUMBER> StencilUtils::names = make_stencil_names();

// =======================================================
// =======================================================
// std::array<std::string,STENCIL_TOTAL_NUMBER> make_valid_solver_names() {

  
// } // make_valid_solver_names

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
