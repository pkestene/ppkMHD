#ifndef MOOD_STENCIL_H_
#define MOOD_STENCIL_H_

#include "shared/kokkos_shared.h"
#include <array>

namespace mood {

/**
 * List of implemented stencil for MOOD numerical schemes, degree is the degree of
 * the multivariate polynomial used to interpolate value in the given stencil.
 *
 * Please note that, for a given degree, the miminal stencil points need to compute
 * the polynomial coefficient (using least squares minimization) is the binomial 
 * number: (dim+degree)! / dim! / degree!
 * 
 * So the stencil size must be "significantly" larger than this number to estimate
 * the polynomial coefficients in a robust manner.
 * 
 * In 2D, the binomial coefficient (2+degree)! / 2! / degree! reduces to
 * (degree+1)(degree+2)/2
 *
 * - 2D - degree 1 (minimum number of points is 2*3/2 = 3 )
 *  x
 * xox
 *  x
 *
 * - 2D - degree 2 (minimum number of points is 3*4/2 = 6)
 * xxx
 * xox
 * xxx
 *
 * - 2D - degree 3 (minimum number of points is 4*5/2 = 10)
 *   x
 *  xxx
 * xxoxx
 *  xxx
 *   x
 *
 * - 2D - degree 3 v2 (minimum number of points is 4*5/2 = 10)
 *   
 * xxxx
 * xxox
 * xxxx
 * xxxx
 *
 * - 2D - degree 4 (minimum number of points is 5*6/2 = 15)
 *   
 * xxxxx
 * xxxxx
 * xxoxx
 * xxxxx
 * xxxxx
 *
 * - 2D - degree 5 (minimum number of points is 6*7/2 = 21)
 *   
 * xxxxxx
 * xxxxxx
 * xxxoxx
 * xxxxxx
 * xxxxxx
 * xxxxxx
 *
 * ---------------------------------------------------------
 *
 * In 3D, the binomial coefficient (3+degree)! / 3! / degree! reduces to
 * (degree+1)(degree+2)(degree+3)/6
 *
 * - 3D - degree 1 (minimum number of points is 2*3*4/6 =  4)
 * - 3D - degree 2 (minimum number of points is 3*4*5/6 = 10)
 * - 3D - degree 3 (minimum number of points is 4*5*6/6 = 20)
 * - 3D - degree 4 (minimum number of points is 5*6*7/6 = 35)
 * - 3D - degree 5 (minimum number of points is 6*7*8/6 = 56)
 * 
 * Last item is not a stencil, just a trick to get the number of stencils.
 */
enum STENCIL_ID {
  STENCIL_2D_DEGREE1,       /*  0 */
  STENCIL_2D_DEGREE2,       /*  1 */
  STENCIL_2D_DEGREE3,       /*  2 */
  STENCIL_2D_DEGREE3_V2,    /*  3 */
  STENCIL_2D_DEGREE4,       /*  4 */
  STENCIL_2D_DEGREE5,       /*  5 */
  STENCIL_3D_DEGREE1,       /*  6 */
  STENCIL_3D_DEGREE2,       /*  7 */
  STENCIL_3D_DEGREE3,       /*  8 */
  STENCIL_3D_DEGREE3_V2,    /*  9 */
  STENCIL_3D_DEGREE4,       /* 10 */
  STENCIL_3D_DEGREE5,       /* 11 */
  STENCIL_3D_DEGREE5_V2,    /* 12 */
  STENCIL_TOTAL_NUMBER /* This is not a stencil ! */
};

/**
 * The following array is a "solution" to the probleme faced when one want
 * to get stencil size at compile in a template class; get_stencil_size routine
 * can't be used there.
 */
constexpr std::array<int,STENCIL_TOTAL_NUMBER> STENCIL_SIZE =
  {5,  /* STENCIL_2D_DEGREE1 */
   9,  /* STENCIL_2D_DEGREE2 */
   13, /* STENCIL_2D_DEGREE3 */
   16, /* STENCIL_2D_DEGREE3_V2 */
   25, /* STENCIL_2D_DEGREE4 */
   36, /* STENCIL_2D_DEGREE5 */
   7,  /* STENCIL_3D_DEGREE1 */
   27, /* STENCIL_3D_DEGREE2 */
   27, /* STENCIL_3D_DEGREE3 */
   33, /* STENCIL_3D_DEGREE3_V2 */
   64, /* STENCIL_3D_DEGREE4 */
   125,/* STENCIL_3D_DEGREE5 */
   88  /* STENCIL_3D_DEGREE5_V2 */
  };

/**
 * Return how many cell is made the stencil.
 *
 * \param[in] stencilId a valid value from enum STENCIL_ID
 *
 * \return number of cells contained in stencil.
 */
KOKKOS_INLINE_FUNCTION
unsigned int get_stencil_size(STENCIL_ID stencilId)  {

  if (stencilId == STENCIL_2D_DEGREE1)
    return 5;
  if (stencilId == STENCIL_2D_DEGREE2)
    return 9;
  if (stencilId == STENCIL_2D_DEGREE3)
    return 13;
  if (stencilId == STENCIL_2D_DEGREE3_V2)
    return 16;
  if (stencilId == STENCIL_2D_DEGREE4)
    return 25;
  if (stencilId == STENCIL_2D_DEGREE5)
    return 36;

  if (stencilId == STENCIL_3D_DEGREE1)
    return 7;
  if (stencilId == STENCIL_3D_DEGREE2)
    return 27;
  if (stencilId == STENCIL_3D_DEGREE3)
    return 27;
  if (stencilId == STENCIL_3D_DEGREE3_V2)
    return 33;
  if (stencilId == STENCIL_3D_DEGREE4)
    return 64;
  if (stencilId == STENCIL_3D_DEGREE5)
    return 125;
  if (stencilId == STENCIL_3D_DEGREE5_V2)
    return 88;

  // we shouldn't be here
  return 0;
  
} // get_stencil_size

/**
 * A constexpr array mapping stencilId to the polynomial degree.
 *
 * Same information as get_stencil_degree, but at compile time.
 */
constexpr std::array<int,STENCIL_TOTAL_NUMBER> STENCIL_DEGREE =
  {1,  /* STENCIL_2D_DEGREE1 */
   2,  /* STENCIL_2D_DEGREE2 */
   3,  /* STENCIL_2D_DEGREE3 */
   3,  /* STENCIL_2D_DEGREE3_V2 */
   4,  /* STENCIL_2D_DEGREE4 */
   5,  /* STENCIL_2D_DEGREE5 */
   1,  /* STENCIL_3D_DEGREE1 */
   2,  /* STENCIL_3D_DEGREE2 */
   3,  /* STENCIL_3D_DEGREE3 */
   3,  /* STENCIL_3D_DEGREE3_V2 */
   4,  /* STENCIL_3D_DEGREE4 */
   5,  /* STENCIL_3D_DEGREE5 */
   5   /* STENCIL_3D_DEGREE5_V2 */
  };

/**
 * Return the polynomial degree used with input stencil.
 *
 * \param[in] stencilId a valid value from enum STENCIL_ID
 *
 * \return polynomial degree.
 */
KOKKOS_INLINE_FUNCTION
unsigned int get_stencil_degree(STENCIL_ID stencilId) {

  if (stencilId == STENCIL_2D_DEGREE1)
    return 1;
  if (stencilId == STENCIL_2D_DEGREE2)
    return 2;
  if (stencilId == STENCIL_2D_DEGREE3)
    return 3;
  if (stencilId == STENCIL_2D_DEGREE3_V2)
    return 3;
  if (stencilId == STENCIL_2D_DEGREE4)
    return 4;
  if (stencilId == STENCIL_2D_DEGREE5)
    return 5;

  if (stencilId == STENCIL_3D_DEGREE1)
    return 1;
  if (stencilId == STENCIL_3D_DEGREE2)
    return 2;
  if (stencilId == STENCIL_3D_DEGREE3)
    return 3;
  if (stencilId == STENCIL_3D_DEGREE3_V2)
    return 3;
  if (stencilId == STENCIL_3D_DEGREE4)
    return 4;
  if (stencilId == STENCIL_3D_DEGREE5)
    return 5;
  if (stencilId == STENCIL_3D_DEGREE5_V2)
    return 5;

  // we shouldn't be here
  return 0;
  
} // get_stencil_degree

/**
 * Return the ghostwidth one should use to define a propoer mood 
 * computational kernel.
 *
 * \param[in] stencilId a valid value from enum STENCIL_ID
 *
 * \return ghostwidth.
 */
KOKKOS_INLINE_FUNCTION
unsigned int get_stencil_ghostwidth(STENCIL_ID stencilId) {

  if (stencilId == STENCIL_2D_DEGREE1)
    return 2;
  if (stencilId == STENCIL_2D_DEGREE2)
    return 2;
  if (stencilId == STENCIL_2D_DEGREE3)
    return 3;
  if (stencilId == STENCIL_2D_DEGREE3_V2)
    return 3;
  if (stencilId == STENCIL_2D_DEGREE4)
    return 3;
  if (stencilId == STENCIL_2D_DEGREE5)
    return 4;

  if (stencilId == STENCIL_3D_DEGREE1)
    return 2;
  if (stencilId == STENCIL_3D_DEGREE2)
    return 2;
  if (stencilId == STENCIL_3D_DEGREE3)
    return 2;
  if (stencilId == STENCIL_3D_DEGREE3_V2)
    return 3;
  if (stencilId == STENCIL_3D_DEGREE4)
    return 3;
  if (stencilId == STENCIL_3D_DEGREE5)
    return 3;
  if (stencilId == STENCIL_3D_DEGREE5_V2)
    return 4;

  // we shouldn't be here
  return 0;
  
} // get_stencil_ghostwidth

/**
 * \struct Stencil
 * Just store the coordinates (x,y,z) of the neighbor cells contained in the stencil.
 */
struct Stencil {

  /**
   * a 2D array data type;
   * first dimension is the neighbor index
   * second dimension is the coordinate index
   */
  using StencilArray     = Kokkos::View<int*[3], DEVICE>;
  using StencilArrayHost = StencilArray::HostMirror;
  
  //! number identify stencil (values from enum STENCIL_ID)
  STENCIL_ID stencilId;

  //! stencil size, is usefull for allocating offsets arrays
  unsigned int stencilSize;

  //! coordinates of the neighbors on DEVICE
  StencilArray offsets;

  //! coordinates of the neighbors on HOST
  StencilArrayHost offsets_h;

  //! is it a 3D stencil ?
  bool is3dStencil;
  
  //! default constructor
  Stencil(STENCIL_ID stencilId) : stencilId(stencilId) {

    // we should check for stencilId here;
    // valid value is in range [0, STENCIL_TOTAL_NUMBER-1]
    
    stencilSize = get_stencil_size(stencilId);

    is3dStencil = (stencilId >= STENCIL_3D_DEGREE1) ? true : false;
    
    offsets   = StencilArray    ("offsets",   stencilSize);

    offsets_h = StencilArrayHost("offsets_h", stencilSize);

    init_stencil();
    
  }; // Stencil

  void init_stencil();

  static STENCIL_ID select_stencil(unsigned int dim, unsigned int degree);
  
}; // Stencil

} // namespace mood

#endif // MOOD_STENCIL_H_
