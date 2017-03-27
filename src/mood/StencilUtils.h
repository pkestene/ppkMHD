#ifndef MOOD_STENCIL_UTILS_H_
#define MOOD_STENCIL_UTILS_H_

#include "mood/Stencil.h"

#include <array>
#include <string>

namespace mood {

/**
 *
 */
struct StencilUtils {

  StencilUtils() {};

  static std::string get_stencil_name(STENCIL_ID id);
  static STENCIL_ID get_stencilId_from_string(const std::string& name);
  static STENCIL_ID get_stencilId_from_solver_name(const std::string& name);
  
  static std::array<std::string,STENCIL_TOTAL_NUMBER> names;
  static std::array<std::string,STENCIL_TOTAL_NUMBER> solver_names;

  static void print_stencil(const Stencil& stencil);
  
};

std::array<std::string,STENCIL_TOTAL_NUMBER> make_stencil_names();
std::array<std::string,STENCIL_TOTAL_NUMBER> make_valid_solver_names();


} // namespace mood

#endif // MOOD_STENCIL_UTILS_H_
