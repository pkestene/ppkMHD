#ifndef IO_VTK_H_
#define IO_VTK_H_

#include <kokkos_shared.h>

namespace ppkMHD { namespace io {

void save_VTK_2D(DataArray2d Udata,
		 int iStep);
void save_VTK_3D(DataArray3d Udata,
		 int iStep);

} // namespace io

} // namespace ppkMHD

#endif // IO_VTK_H_
