/*
 * g++ -Wall test_lapacke.cpp -o test_lapacke -llapacke
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <lapacke.h>

/* Auxiliary routine: printing a matrix */
void print_matrix_rowmajor( char* desc,
			    lapack_int m,
			    lapack_int n,
			    double* mat,
			    lapack_int ldm ) {
  lapack_int i, j;
  printf( "\n %s\n", desc );
  
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < n; j++ ) printf( " %6.2f", mat[i*ldm+j] );
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
  lapack_int n, nrhs, lda, ldb, info;
  int i, j;
  
  /* Local arrays */
  double *A, *b;
  lapack_int *ipiv;
  
  /* Default Value */
  n = 5; nrhs = 1;
  
  /* Arguments */
  for( i = 1; i < argc; i++ ) {
    if( strcmp( argv[i], "-n" ) == 0 ) {
      n  = atoi(argv[i+1]);
      i++;
    }
    if( strcmp( argv[i], "-nrhs" ) == 0 ) {
      nrhs  = atoi(argv[i+1]);
      i++;
    }
  }
  
  /* Initialization */
  lda=n, ldb=nrhs;
  A = (double *)malloc(n*n*sizeof(double)) ;
  if (A==NULL){ printf("error of memory allocation\n"); exit(0); }
  b = (double *)malloc(n*nrhs*sizeof(double)) ;
  if (b==NULL){ printf("error of memory allocation\n"); exit(0); }
  ipiv = (lapack_int *)malloc(n*sizeof(lapack_int)) ;
  if (ipiv==NULL){ printf("error of memory allocation\n"); exit(0); }
  
  for( i = 0; i < n; i++ ) {
    for( j = 0; j < n; j++ ) A[i*lda+j] = ((double) rand()) / ((double) RAND_MAX) - 0.5;
  }
  
  for(i=0;i<n*nrhs;i++)
    b[i] = ((double) rand()) / ((double) RAND_MAX) - 0.5;
  
  /* Print Entry Matrix */
  print_matrix_rowmajor( "Entry Matrix A", n, n, A, lda );
  /* Print Right Rand Side */
  print_matrix_rowmajor( "Right Rand Side b", n, nrhs, b, ldb );
  printf( "\n" );
  /* Executable statements */
  printf( "LAPACKE_dgesv (row-major, high-level) Example Program Results\n" );
  /* Solve the equations A*X = B */
  info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, nrhs, A, lda, ipiv,
                        b, ldb );
  /* Check for the exact singularity */
  if( info > 0 ) {
    printf( "The diagonal element of the triangular factor of A,\n" );
    printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
    printf( "the solution could not be computed.\n" );
    exit( 1 );
  }
  if (info <0) exit( 1 );
  /* Print solution */
  print_matrix_rowmajor( "Solution", n, nrhs, b, ldb );
  /* Print details of LU factorization */
  print_matrix_rowmajor( "Details of LU factorization", n, n, A, lda );
  /* Print pivot indices */
  print_vector( "Pivot indices", n, ipiv );
  exit( EXIT_SUCCESS );
	
} // End of LAPACKE_dgeqrf Example
