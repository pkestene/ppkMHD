#include "mood/Stencil.h"

#include "shared/enums.h"

#include <iostream>

namespace mood {

// =======================================================
// =======================================================
unsigned int get_stencil_size(STENCIL_ID stencilId) {

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

} // get_stencil_size

// =======================================================
// =======================================================
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

} // get_stencil_degree

// =======================================================
// ==== STRUCT Stencil IMPL ==============================
// =======================================================

// =======================================================
// =======================================================
void Stencil::init_stencil()
{

  //std::cout << "Initializing stencil " << stencilId << "\n";

  // ------------------------------------
  // ------------------------------------
  if (stencilId == STENCIL_2D_DEGREE1) {

    offsets_h(0,IX) = 0;
    offsets_h(0,IY) = 0;
    offsets_h(0,IZ) = 0;

    offsets_h(1,IX) = 1;
    offsets_h(1,IY) = 0;
    offsets_h(1,IZ) = 0;

    offsets_h(2,IX) = 0;
    offsets_h(2,IY) = 1;
    offsets_h(2,IZ) = 0;

    offsets_h(3,IX) = -1;
    offsets_h(3,IY) = 0;
    offsets_h(3,IZ) = 0;

    offsets_h(4,IX) = 0;
    offsets_h(4,IY) = -1;
    offsets_h(4,IZ) = 0;

  } // STENCIL_2D_DEGREE1
  
  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_2D_DEGREE2) {

    // 3-by-3 = 9 points
    
    int index = 0;
    for (int j = -1; j <= 1; ++j) {
      for (int i = -1; i <= 1; ++i) {
	offsets_h(index,IX) = i;
	offsets_h(index,IY) = j;
	offsets_h(index,IZ) = 0;
	index++;
      }
    }

  } // STENCIL_2D_DEGREE2
    
  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_2D_DEGREE3) {

    // same as degree2 + add "external" points
    // 9+4=13 points
    
    int index = 0;
    for (int j = -1; j <= 1; ++j) {
      for (int i = -1; i <= 1; ++i) {
	offsets_h(index,IX) = i;
	offsets_h(index,IY) = j;
	offsets_h(index,IZ) = 0;
	index++;
      }
    }

    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;
    
    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = 0; index++;
    
    offsets_h(index,IX) = -2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;
    
    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -2;
    offsets_h(index,IZ) = 0; index++;
    
  } // STENCIL_2D_DEGREE3
    
  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_2D_DEGREE3_V2) {

    // 4-by-4 = 16 points
    
    int index = 0;
    for (int j = -2; j <= 1; ++j) {
      for (int i = -2; i <= 1; ++i) {
	offsets_h(index,IX) = i;
	offsets_h(index,IY) = j;
	offsets_h(index,IZ) = 0;
	index++;
      }
    }

  } // STENCIL_2D_DEGREE3_V2

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_2D_DEGREE4) {

    // 5-by-5 = 25 points
    
    int index = 0;
    for (int j = -2; j <= 2; ++j) {
      for (int i = -2; i <= 2; ++i) {
	offsets_h(index,IX) = i;
	offsets_h(index,IY) = j;
	offsets_h(index,IZ) = 0;
	index++;
      }
    }

  } // STENCIL_2D_DEGREE4

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_2D_DEGREE5) {

    // 6-by-6 = 36 points
    
    int index = 0;
    for (int j = -3; j <= 2; ++j) {
      for (int i = -3; i <= 2; ++i) {
	offsets_h(index,IX) = i;
	offsets_h(index,IY) = j;
	offsets_h(index,IZ) = 0;
	index++;
      }
    }

  } // STENCIL_2D_DEGREE5


  // ------------------------------------
  // ------------------------------------
  // 3D
  // ------------------------------------
  // ------------------------------------

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE1) {

    offsets_h(0,IX) = 0;
    offsets_h(0,IY) = 0;
    offsets_h(0,IZ) = 0;

    offsets_h(1,IX) = 1;
    offsets_h(1,IY) = 0;
    offsets_h(1,IZ) = 0;

    offsets_h(2,IX) = 0;
    offsets_h(2,IY) = 1;
    offsets_h(2,IZ) = 0;

    offsets_h(3,IX) = -1;
    offsets_h(3,IY) = 0;
    offsets_h(3,IZ) = 0;

    offsets_h(4,IX) = 0;
    offsets_h(4,IY) = -1;
    offsets_h(4,IZ) = 0;

    offsets_h(5,IX) = 0;
    offsets_h(5,IY) = 0;
    offsets_h(5,IZ) = 1;

    offsets_h(6,IX) = 0;
    offsets_h(6,IY) = 0;
    offsets_h(6,IZ) = -1;

  } // STENCIL_3D_DEGREE1
  
  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE2 || stencilId == STENCIL_3D_DEGREE3) {

    // 3-by-3-by-3 = 27 points
    
    int index = 0;
    for (int k = -1; k <= 1; ++k) {
      for (int j = -1; j <= 1; ++j) {
	for (int i = -1; i <= 1; ++i) {
	  offsets_h(index,IX) = i;
	  offsets_h(index,IY) = j;
	  offsets_h(index,IZ) = k;
	  index++;
	}
      }
    }

  } // STENCIL_3D_DEGREE2 || STENCIL_3D_DEGREE3
  
  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE3_V2) {

    // 3-by-3-by-3 + 6 = 33 points
    
    int index = 0;
    for (int k = -1; k <= 1; ++k) {
      for (int j = -1; j <= 1; ++j) {
	for (int i = -1; i <= 1; ++i) {
	  offsets_h(index,IX) = i;
	  offsets_h(index,IY) = j;
	  offsets_h(index,IZ) = k;
	  index++;
	}
      }
    }

    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;
    offsets_h(index,IX) = -2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = 0; index++;
    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -2;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 2; index++;
    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = -2; index++;

  } // STENCIL_3D_DEGREE3_V2

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE4) {

    // 4-by-4-by-4 = 64 points
    
    int index = 0;
    for (int k = -2; k <= 1; ++k) {
      for (int j = -2; j <= 1; ++j) {
	for (int i = -2; i <= 1; ++i) {
	  offsets_h(index,IX) = i;
	  offsets_h(index,IY) = j;
	  offsets_h(index,IZ) = k;
	  index++;
	}
      }
    }

  } // STENCIL_3D_DEGREE4

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE5) {

    // 5-by-5-by-5 = 125 points
    
    int index = 0;
    for (int k = -2; k <= 2; ++k) {
      for (int j = -2; j <= 2; ++j) {
	for (int i = -2; i <= 2; ++i) {
	  offsets_h(index,IX) = i;
	  offsets_h(index,IY) = j;
	  offsets_h(index,IZ) = k;
	  index++;
	}
      }
    }

  } // STENCIL_3D_DEGREE5

  // ------------------------------------
  // ------------------------------------
  else if (stencilId == STENCIL_3D_DEGREE5_V2) {

    // 4-by-4-by-4 = 64 points + 6*4 = 88 points
    
    int index = 0;
    for (int k = -2; k <= 1; ++k) {
      for (int j = -2; j <= 1; ++j) {
	for (int i = -2; i <= 1; ++i) {
	  offsets_h(index,IX) = i;
	  offsets_h(index,IY) = j;
	  offsets_h(index,IZ) = k;
	  index++;
	}
      }
    }

    // face +x
    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = 2;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;

    // face -x
    offsets_h(index,IX) = -3;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = -3;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = -3;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = -3;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 0; index++;

    // face +y
    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 2;
    offsets_h(index,IZ) = 0; index++;
    
    // face -y
    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = -3;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = -3;
    offsets_h(index,IZ) = 0; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -3;
    offsets_h(index,IZ) = -1; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -3;
    offsets_h(index,IZ) = 0; index++;
    
    // face +z
    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = 2; index++;

    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 2; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = 2; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = 2; index++;
    
    // face -z
    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = -3; index++;

    offsets_h(index,IX) = -1;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = -3; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = -1;
    offsets_h(index,IZ) = -3; index++;

    offsets_h(index,IX) = 0;
    offsets_h(index,IY) = 0;
    offsets_h(index,IZ) = -3; index++;
    
  } // STENCIL_3D_DEGREE5

  // upload data to DEVICE
  Kokkos::deep_copy(offsets, offsets_h);
  
} // Stencil::init_stencil

// =======================================================
// =======================================================
STENCIL_ID Stencil::select_stencil(unsigned int dim, unsigned int degree)
{

  int res=STENCIL_2D_DEGREE1;

  if (dim < 2 or dim > 3) {
    std::cerr << "Only 2D / 3D are allowed\n";
    std::cerr << "Returning default value, i.e. STENCIL_2D_DEGREE1\n";
  }

  if (degree<1 or degree>6) {
    std::cerr << "Not implemented !\n";
    std::cerr << "polynomial degree must be >=1 or <=5.\n";
    std::cerr << "Returning default value, i.e. STENCIL_2D_DEGREE1\n";
  }

  if (dim==2) {
    
    if        (degree==1) {
      return STENCIL_2D_DEGREE1;
    } else if (degree==2) {
      return STENCIL_2D_DEGREE2;
    } else if (degree==3) {
      return STENCIL_2D_DEGREE3;
    } else if (degree==4) {
      return STENCIL_2D_DEGREE4;
    } else if (degree==5) {
      return STENCIL_2D_DEGREE5;
    }
    
  } else if (dim==3) {

    if        (degree==1) {
      return STENCIL_3D_DEGREE1;
    } else if (degree==2) {
      return STENCIL_3D_DEGREE2;
    } else if (degree==3) {
      return STENCIL_3D_DEGREE3;
    } else if (degree==4) {
      return STENCIL_3D_DEGREE4;
    } else if (degree==5) {
      return STENCIL_3D_DEGREE5;
    }

  }
  
} // Stencil::select_stencil

} // namespace mood

