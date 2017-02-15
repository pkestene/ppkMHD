/**
 * Adapted / simplified from boost::mpi environment.hpp
 * Meant to be easier to build with compiler like nvcc.
 *
 *
 *  This header provides the @c environment class, which provides
 *  routines to initialize, finalization, and query the status of the
 *  Boost MPI environment.
 *
 */

// Copyright (C) 2006 Douglas Gregor <doug.gregor -at- gmail.com>

// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef MPI_ENV_H_
#define MPI_ENV_H_

/* Force MPICH not to define SEEK_SET, SEEK_CUR, and SEEK_END, which
   conflict with the versions in <stdio.h> and <cstdio>. */
#define MPICH_IGNORE_CXX_SEEK 1

#include <mpi.h>

#include <string>
#include <iosfwd>

namespace ppkMHD { namespace mpi {

namespace threading {

/** @brief specify the supported threading level.
 * 
 * Based on MPI 2 standard/8.7.3
 */
enum level {
  /** Only one thread will execute. 
   */
  single     = MPI_THREAD_SINGLE,
  /** Only main thread will do MPI calls.
   * 
   * The process may be multi-threaded, but only the main 
   * thread will make MPI calls (all MPI calls are ``funneled''
   * to the main thread).
   */
  funneled   = MPI_THREAD_FUNNELED,
  /** Only one thread at the time do MPI calls.
   * 
   * The process may be multi-threaded, and multiple 
   * threads may make MPI calls, but only one at a time:
   * MPI calls are not made concurrently from two distinct 
   * threads (all MPI calls are ``serialized'').
   */
  serialized = MPI_THREAD_SERIALIZED,
  /** Multiple thread may do MPI calls.
   * 
   * Multiple threads may call MPI, with no restrictions.
   */
  multiple   = MPI_THREAD_MULTIPLE
};

/** Formated output for threading level. */
std::ostream& operator<<(std::ostream& out, level l);

/** Formated input for threading level. */
std::istream& operator>>(std::istream& in, level& l);
} // namespace threading

/** @brief Initialize, finalize, and query the MPI environment.
 *
 *  The @c environment class is used to initialize, finalize, and
 *  query the MPI environment. It will typically be used in the @c
 *  main() function of a program, which will create a single instance
 *  of @c environment initialized with the arguments passed to the
 *  program:
 *
 *  @code
 *  int main(int argc, char* argv[])
 *  {
 *    mpi::environment env(argc, argv);
 *  }
 *  @endcode
 *
 *  The instance of @c environment will initialize MPI (by calling @c
 *  MPI_Init) in its constructor and finalize MPI (by calling @c
 *  MPI_Finalize for normal termination or @c MPI_Abort for an
 *  uncaught exception) in its destructor.
 *
 *  The use of @c environment is not mandatory. Users may choose to
 *  invoke @c MPI_Init and @c MPI_Finalize manually. In this case, no
 *  @c environment object is needed. If one is created, however, it
 *  will do nothing on either construction or destruction.
 */
class environment {

public:

  /** Initialize the MPI environment.
   *
   *  If the MPI environment has not already been initialized,
   *  initializes MPI with a call to @c MPI_Init.
   *
   *  @param argc The number of arguments provided in @p argv, as
   *  passed into the program's @c main function.
   *
   *  @param argv The array of argument strings passed to the program
   *  via @c main.
   *
   *  @param abort_on_exception When true, this object will abort the
   *  program if it is destructed due to an uncaught exception.
   */
  environment(int& argc, char** &argv, bool abort_on_exception = true);

  /** Initialize the MPI environment.
   *
   *  If the MPI environment has not already been initialized,
   *  initializes MPI with a call to @c MPI_Init_thread.
   *
   *  @param argc The number of arguments provided in @p argv, as
   *  passed into the program's @c main function.
   *
   *  @param argv The array of argument strings passed to the program
   *  via @c main.
   *
   *  @param mt_level the required level of threading support
   *
   *  @param abort_on_exception When true, this object will abort the
   *  program if it is destructed due to an uncaught exception.
   */
  environment(int& argc, char** &argv, threading::level mt_level,
              bool abort_on_exception = true);

  /** Shuts down the MPI environment.
   *
   *  If this @c environment object was used to initialize the MPI
   *  environment, and the MPI environment has not already been shut
   *  down (finalized), this destructor will shut down the MPI
   *  environment. Under normal circumstances, this only involves
   *  invoking @c MPI_Finalize. However, if destruction is the result
   *  of an uncaught exception and the @c abort_on_exception parameter
   *  of the constructor had the value @c true, this destructor will
   *  invoke @c MPI_Abort with @c MPI_COMM_WORLD to abort the entire
   *  MPI program with a result code of -1.
   */
  ~environment();

  /** Abort all MPI processes.
   *
   *  Aborts all MPI processes and returns to the environment. The
   *  precise behavior will be defined by the underlying MPI
   *  implementation. This is equivalent to a call to @c MPI_Abort
   *  with @c MPI_COMM_WORLD.
   *
   *  @param errcode The error code to return to the environment.
   *  @returns Will not return.
   */
  static void abort(int errcode);

  /** Determine if the MPI environment has already been initialized.
   *
   *  This routine is equivalent to a call to @c MPI_Initialized.
   *
   *  @returns @c true if the MPI environment has been initialized.
   */
  static bool initialized();

  /** Determine if the MPI environment has already been finalized.
   *
   *  The routine is equivalent to a call to @c MPI_Finalized.
   *
   *  @returns @c true if the MPI environment has been finalized.
   */
  static bool finalized();

  /** Retrieves the maximum tag value.
   *
   *  Returns the maximum value that may be used for the @c tag
   *  parameter of send/receive operations. This value will be
   *  somewhat smaller than the value of @c MPI_TAG_UB, because the
   *  Boost.MPI implementation reserves some tags for collective
   *  operations.
   *
   *  @returns the maximum tag value.
   */
  static int max_tag();

  /** The tag value used for collective operations.
   *
   *  Returns the reserved tag value used by the Boost.MPI
   *  implementation for collective operations. Although users are not
   *  permitted to use this tag to send or receive messages, it may be
   *  useful when monitoring communication patterns.
   *
   * @returns the tag value used for collective operations.
   */
  static int collectives_tag();

  /** Retrieve the name of this processor.
   *
   *  This routine returns the name of this processor. The actual form
   *  of the name is unspecified, but may be documented by the
   *  underlying MPI implementation. This routine is implemented as a
   *  call to @c MPI_Get_processor_name.
   *
   *  @returns the name of this processor.
   */
  static std::string processor_name();

  /** Query the current level of thread support.
   */
  static threading::level thread_level();

  /** Are we in the main thread?
   */
  static bool is_main_thread();
  
  /** @brief MPI version.
   *
   * Returns a pair with the version and sub-version number.
   */
  static std::pair<int, int> version();

private:
  /// Whether this environment object called MPI_Init
  bool i_initialized;

  /// Whether we should abort if the destructor is
  bool abort_on_exception;
  
  /// The number of reserved tags.
  static const int num_reserved_tags = 1;
};

} } // end namespace ppkMHD::mpi

#endif // MPI_ENV_H_
