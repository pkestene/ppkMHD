#ifndef STENCIL_UTILS_H_
#define STENCIL_UTILS_H_

#include "mood/Stencil.h"

#include <array>
#include <string>

namespace mood {

/**
 *
 */
struct StencilUtils {

  StencilUtils();

  static std::string get_stencil_name(STENCIL_ID id);

  static std::array<std::string,STENCIL_TOTAL_NUMBER> names;

  static void print_stencil(const Stencil& stencil);
  
};

std::array<std::string,STENCIL_TOTAL_NUMBER> make_stencil_names();


} // namespace mood

#endif // STENCIL_UTILS_H_
