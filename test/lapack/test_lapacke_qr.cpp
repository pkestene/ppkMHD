/*
 * g++ -Wall test_lapacke_qr.cpp -o test_lapacke_qr -llapacke -lblas
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lapacke.h>
#include <cblas.h>

/* Auxiliary routine: printing a matrix */
void print_matrix_rowmajor( char* desc,
			    lapack_int m,
			    lapack_int n,
			    double* mat,
			    lapack_int ldm ) {
  lapack_int i, j;
  printf( "\n %s\n", desc );
  
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) printf( " %6.5f", mat[i*ldm+j] );
    printf( "\n" );
  }
}

/* Auxiliary routine: printing a vector of integers */
void print_vector( char* desc, lapack_int n, lapack_int* vec ) {
        lapack_int j;
        printf( "\n %s\n", desc );
        for( j = 0; j < n; j++ ) printf( " %6i", vec[j] );
        printf( "\n" );
}

/* ################################################# */
/* ################################################# */
/* ################################################# */

/* Main program */
int main(int argc, char *argv[])
{
  
  /* Locals */
  lapack_int m, n, lda, ldat, lda_check, info;
  int i, j, k;

  lapack_int min_mn;
  
  /* Local arrays */
  double *A; /* size m,n (m>n) */
  double *A_cpy; /* size m,n (m>n) */
  double *AT; /* size n,m transpose */
  double *Q; /* size m,m */
  double *R; /* size m,n */

  double *Check; /* size n,n */
  
  double *tau;
  
  /* Default Value */
  m = 5;
  n = 3;

  min_mn = m<n ? m : n;
  
  /* Initialization */
  lda=n;
  ldat=m;
  lda_check=n;
  
  A = (double *) malloc(m*n*sizeof(double)) ;
  if (A==NULL){ printf("error of memory allocation\n"); exit(0); }

  A_cpy = (double *) malloc(m*n*sizeof(double)) ;
  if (A_cpy==NULL){ printf("error of memory allocation\n"); exit(0); }

  AT = (double *) malloc(n*m*sizeof(double)) ;
  if (AT==NULL){ printf("error of memory allocation\n"); exit(0); }

  Q = (double *) malloc(m*m*sizeof(double)) ;
  if (Q==NULL){ printf("error of memory allocation\n"); exit(0); }

  R = (double *) malloc(m*n*sizeof(double)) ;
  if (R==NULL){ printf("error of memory allocation\n"); exit(0); }

  tau = (double*) malloc(min_mn*sizeof(double));

  Check = (double *) malloc(n*n*sizeof(double)) ;
  if (Check==NULL){ printf("error of memory allocation\n"); exit(0); }
  
  // for( i = 0; i < n; i++ ) {
  //   for( j = 0; j < n; j++ )
  //     A[i*lda+j] = ((double) rand()) / ((double) RAND_MAX) - 0.5;
  // }

  A[ 0] = 12; A[ 1] = -51; A[ 2] =  4;
  A[ 3] =  6; A[ 4] = 167; A[ 5] =-68;
  A[ 6] = -4; A[ 7] =  24; A[ 8] =-41;
  A[ 9] = -1; A[10] =   1; A[11] =  0;
  A[12] =  2; A[13] =   0; A[14] =  3;

  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) {
      A_cpy[i*lda+j]  = A[i*lda+j];
      AT   [j*ldat+i] = A[i*lda+j];
    }
  }
  
  for(i=0;i<m*m;i++)
    Q[i] = 0;
  for(i=0;i<m*n;i++)
    R[i] = 0;
  
  /* Print Entry Matrix */
  print_matrix_rowmajor( "Entry Matrix A", m, n, A, lda );
  print_matrix_rowmajor( "Entry Matrix AT", n, m, AT, ldat );
  printf( "\n" );

  // doc lapacke
  // 
  

  /* QR decomposition */
  printf( "LAPACKE_dgeqrf (row-major, high-level)\n" );
  info = LAPACKE_dgeqrf( LAPACK_ROW_MAJOR, m, n, A, lda, tau );
  printf("LAPACKE_dgeqrf status: %d\n",info);

  /* Print pseudo-inverse */
  print_matrix_rowmajor( "Entry R", m, n, A, lda );

  // doc cblas
  // http://www.netlib.org/lapack/explore-html/d2/d24/cblas__dtrsm_8c_source.html
  // https://software.intel.com/en-us/node/521004
  
  // http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf
  // pseudo-inverse is A_dag = (AT*A)^-1 * AT
  
  // solve R^T * X = A^T, i.e. compute X = (R^T)^(-1) * A^T,
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
  	      n, m, 1.0, A, lda, AT, ldat);

  print_matrix_rowmajor( "Entry (R^T)^(-1) * A^T", n, m, AT, ldat );

  
  // solve R * X=(R^T)^-1*A^T, i.e compute X = (R^-1)*(R^T)^(-1) * A^T
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
	      n, m, 1.0, A, lda, AT, ldat);

  /* Print pseudo-inverse */
  print_matrix_rowmajor( "Entry Pseudo-inverse", n, m, AT, ldat );

  /* compute Check = A_PI * A */
  for( i = 0; i < n; i++ ) {
    for( j = 0; j < n; j++ ) {
      Check[i*lda_check+j] = 0;
    }
  }

  for( i = 0; i < n; i++ ) {
    for( j = 0; j < n; j++ ) {
      for( k = 0; k < m; k++ ) {
	Check[i*lda_check+j] += AT[i*ldat+k] * A_cpy[k*lda+j];
      }
    }
  }

  print_matrix_rowmajor( "Entry Check", n, n, Check, lda_check );

  exit( EXIT_SUCCESS );
	
} // End of LAPACKE_dgeqrf Example
