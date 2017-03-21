#ifndef MOOD_MATRIX_H_
#define MOOD_MATRIX_H_

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm> // for std::copy
#include <iostream>

namespace mood {

// enum LAPACKE_TYPE_MAJOR {
//   LAPACKE_ROW_MAJOR,
//   LAPACKE_COL_MAJOR
// };

/**
 * A simple matrix class that will be used to interact with Lapacke 
 * (C-interface to lapacke.
 */
class Matrix {

public:
  
  //! default constructor (don't allocate)
  Matrix() : m(0), n(0), _data(nullptr) {}
  
  //! constructor with memory allocation, initialized to zero
  Matrix(int m_, int n_) : Matrix() {
    m = m_;
    n = n_;
    allocate(m_,n_);
  }

  //! copy constructor
  Matrix(const Matrix& mat) : Matrix(mat.m,mat.n) {

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
	(*this)(i,j) = mat(i,j);
  }
  
  //! constructor from array
  template<int rows, int cols>
  Matrix(double (&a)[rows][cols]) : Matrix(rows,cols) {

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
	(*this)(i,j) = a[i][j];
  }

  //! destructor
  ~Matrix() {
    deallocate();
  }


  /**
   * Access data operators.
   *
   * Data accessor use row major-order by default.
   * (Should probably provide a template parameter for Matrix
   * to make it generic).
   */
  double& operator() (int i, int j) {
    return _data[i*n+j]; }
  double  operator() (int i, int j) const {
    return _data[i*n+j]; }

  double* data()             { return _data; }
  const double* data() const { return _data; }
  
  //! operator assignment
  Matrix& operator=(const Matrix& source) {
    
    // self-assignment check
    if (this != &source) { 
      if ( (m*n) != (source.m * source.n) ) { // storage cannot be reused
	allocate(source.m,source.n);          // re-allocate storage
      }
      // storage can be used, copy data
      std::copy(source.data(), source.data() + source.m*source.n, _data);
    }
    return *this;
  } // operator=
  
  //! compute minor (in place).
  void compute_minor(const Matrix& mat, int d) {

    allocate(mat.m, mat.n);
    
    for (int i = 0; i < d; i++)
      (*this)(i,i) = 1.0;
    for (int i = d; i < mat.m; i++)
      for (int j = d; j < mat.n; j++)
	(*this)(i,j) = mat(i,j);
    
  } // compute_minor

  /**
   * Matrix multiplication.
   * this = a * b
   * this will be re-allocated here
   *
   * \param[in] a left factor Matrix
   * \param[in] b right factor Matrix
   */
  void mult(const Matrix& a, const Matrix& b) {

    if (a.n != b.m) {
      std::cerr << "Matrix multiplication not possible, sizes don't match !\n";
      return;
    }

    // reallocate ourself if necessary i.e. current Matrix has not valid sizes
    if (a.m != m or b.n != n)
      allocate(a.m, b.n);

    memset(_data,0,m*n*sizeof(double));
    
    for (int i = 0; i < a.m; i++)
      for (int j = 0; j < b.n; j++)
	for (int k = 0; k < a.n; k++)
	  (*this)(i,j) += a(i,k) * b(k,j);
    
  } // mult

  //! Perform matrix transposition in-place (for square matrices only).
  void transpose() {
    if (m==n) {
      for (int i = 0; i < m; i++) {
	for (int j = 0; j < i; j++) {
	  double t = (*this)(i,j);
	  (*this)(i,j) = (*this)(j,i);
	  (*this)(j,i) = t;
	}
      }
    } else {
      double* tmp = (double *) malloc(m*n*sizeof(double));
      memcpy(tmp,_data,m*n*sizeof(double));
      for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++) {
	  _data[j*m+i] = tmp[i*n+j];
	}
      }
      std::swap(m,n);
    }
  } // transpose

  //! memory allocation
  void allocate(int m_, int n_) {

    // if already allocated, memory is freed
    deallocate();
    
    // new sizes
    m = m_;
    n = n_;
    
    _data = new double[m_*n_];
    memset(_data,0,m_*n_*sizeof(double));

  } // allocate

  //! memory free
  void deallocate() {

    if (_data)
      delete[] _data;

    _data = nullptr;

  } // deallocate

  //! print on stdout
  void print(const char* desc) {

    printf( "\n %s\n", desc );
    
    for ( int i = 0; i < m; ++i ) {
      for ( int j = 0; j < n; ++j ) {
	printf( "% 12.5f ", (*this)(i,j) );
      }
      printf( "\n" );
    }

  } // print

  //! reset matrix, all entries set to zero.
  void reset() {memset((void*) _data, 0, m*n*sizeof(double));  };

  /** 
   * QR decomposition using LAPACKE + BLAS.
   *
   * this = Q * R, where Q is orthogonal and R is upper triangular.
   *
   * This is just a wrapper to LAPACKE_dgeqrf.
   * After this call, this->data will contain contain the upper triangular
   * matrix and householder coefficients. See
   */
  void compute_qr();

  //! leading dimension (row-major)
  int ld() {return n;}
  
  //! matrix sizes (m is number of rows, n is number of columns)
  int m, n;

  
private:
  
  //! matrix data array
  double* _data;
  
}; // class Matrix

/**
 * Compute pseudo-inverse of a Matrix using its QR decompostion.
 *
 * \param[in] A input matrix
 * \param[out] A_PI pseudo-inverse matrix of A
 *
 * See doc 
 * http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf
 * pseudo-inverse is A_dag = (A^T*A)^-1 * A^T
 *
 */ 
void compute_pseudo_inverse(const Matrix& A, Matrix& A_PI);

} // namespace mood

#endif // MOOD_MATRIX_H_

