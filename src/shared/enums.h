#ifndef SHARED_ENUMS_H_
#define SHARED_ENUMS_H_

//! dimension of the problem
enum DimensionType {
  TWO_D = 2, 
  THREE_D = 3,
  DIM2 = 2,
  DIM3 = 3
};

//! hydro field indexes
enum VarIndex {
  ID=0,   /*!< ID Density field index */
  IP=1,   /*!< IP Pressure/Energy field index */
  IU=2,   /*!< X velocity / momentum index */
  IV=3,   /*!< Y velocity / momentum index */ 
  IW=4,   /*!< Z velocity / momentum index */ 
  IA=5,   /*!< X magnetic field index */ 
  IB=6,   /*!< Y magnetic field index */ 
  IC=7,   /*!< Z magnetic field index */ 
  IBX=5,  /*!< X magnetic field index */ 
  IBY=6,  /*!< Y magnetic field index */ 
  IBZ=7,   /*!< Z magnetic field index */
  IBFX = 0,
  IBFY = 1,
  IBFZ = 2
};

//! face index
enum FaceIdType {
  FACE_XMIN=0,
  FACE_XMAX=1,
  FACE_YMIN=2,
  FACE_YMAX=3,
  FACE_ZMIN=4,
  FACE_ZMAX=5,
  FACE_MIN =0,
  FACE_MAX =1
};

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_APPROX, /*!< quasi-exact Riemann solver (hydro-only) */ 
  RIEMANN_HLL,    /*!< HLL hydro and MHD Riemann solver */
  RIEMANN_HLLC,   /*!< HLLC hydro-only Riemann solver */
  RIEMANN_HLLD    /*!< HLLD MHD-only Riemann solver */
};

//! type of boundary condition (note that BC_COPY is only used in the
//! MPI version for inside boundary)
enum BoundaryConditionType {
  BC_UNDEFINED, 
  BC_DIRICHLET,   /*!< reflecting border condition */
  BC_NEUMANN,     /*!< absorbing border condition */
  BC_PERIODIC,    /*!< periodic border condition */
  BC_COPY         /*!< only used in MPI parallelized version */
};

//! enum component index
enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};

//! direction used in directional splitting scheme
enum Direction {
  XDIR=1, 
  YDIR=2,
  ZDIR=3,
  DIR_X = 0,
  DIR_Y = 1,
  DIR_2 = 2
};

//! location of the outside boundary
enum BoundaryLocation {
  XMIN = 0, 
  XMAX = 1, 
  YMIN = 2, 
  YMAX = 3,
  ZMIN = 4,
  ZMAX = 5
};

//! enum edge index (use in MHD - EMF computations)
enum EdgeIndex {
  IRT = 0, /*!< RT (Right - Top   ) */
  IRB = 1, /*!< RB (Right - Bottom) */
  ILT = 2, /*!< LT (Left  - Top   ) */
  ILB = 3  /*!< LB (Left  - Bottom) */
};

enum EdgeIndex2 {
  ILL = 0,
  IRL = 1,
  ILR = 2,
  IRR = 3
};

//! enum used in MHD - EMF computations
enum EmfDir {
  EMFX = 0,
  EMFY = 1,
  EMFZ = 2
};

//! EMF indexes (EMFZ is first because in 2D, we only need EMFZ)
enum EmfIndex {
  I_EMFZ=0,
  I_EMFY=1,
  I_EMFX=2
};

//! implementation version
enum ImplementationVersion {
  IMPL_VERSION_0,
  IMPL_VERSION_1,
  IMPL_VERSION_2
};

//! problem type
enum ProblemType {
  PROBLEM_IMPLODE,
  PROBLEM_BLAST,
  PROBLEM_ORSZAG_TANG
};

#endif // SHARED_ENUMS_H_
