! roofline_performance_suite_nvfortran.f90
! Advanced CPU Performance Analysis Tool - NVFORTRAN Compatible Version
! Fixes for NVFORTRAN: removed aligned clause, fixed integer types, restructured contains
! 
! Compile with:
!   nvfortran -O3 -mp=multicore -Minfo=all roofline_performance_suite_nvfortran.f90 -o roofline_suite
!   gfortran  -O3 -fopenmp -march=native roofline_performance_suite_nvfortran.f90 -o roofline_suite
!   ifx       -O3 -qopenmp -xHost roofline_performance_suite_nvfortran.f90 -o roofline_suite

module constants_mod
  use iso_fortran_env, only: real64, int64, int32
  implicit none
  
  ! Memory alignment for vectorization
  integer, parameter :: ALIGNMENT = 64
  
  ! Tolerance for validation
  real(real64), parameter :: VALIDATION_TOL = 1.0e-12
  
  ! Kernel identifiers
  integer, parameter :: KERNEL_AXPY = 1
  integer, parameter :: KERNEL_TRIAD = 2
  integer, parameter :: KERNEL_UPDATE3 = 3
  integer, parameter :: KERNEL_DOT = 4
  integer, parameter :: KERNEL_DGEMV = 5
  integer, parameter :: KERNEL_STENCIL5 = 6
  integer, parameter :: KERNEL_ALL = 99
  
end module constants_mod

module timing_mod
  use iso_fortran_env, only: real64, int64, int32
  use omp_lib
  implicit none
  
  type :: perf_stats
    real(real64) :: min_time
    real(real64) :: max_time
    real(real64) :: mean_time
    real(real64) :: median_time
    real(real64) :: stddev_time
    real(real64) :: gflops_mean
    real(real64) :: gflops_peak
    real(real64) :: bandwidth_mean
    real(real64) :: bandwidth_peak
    integer :: num_runs
  end type perf_stats
  
contains

  real(real64) function walltime()
    walltime = omp_get_wtime()
  end function walltime

  subroutine fence_io()
    call flush(6)
  end subroutine fence_io
  
  subroutine compute_statistics(times, n, stats)
    real(real64), intent(in) :: times(:)
    integer, intent(in) :: n
    type(perf_stats), intent(out) :: stats
    real(real64), allocatable :: sorted_times(:)
    real(real64) :: sum_val, sum_sq
    integer :: i
    
    allocate(sorted_times(n))
    sorted_times = times(1:n)
    call quicksort(sorted_times, 1_int32, int(n, int32))
    
    stats%num_runs = n
    stats%min_time = sorted_times(1)
    stats%max_time = sorted_times(n)
    
    ! Mean
    sum_val = 0.0_real64
    do i = 1, n
      sum_val = sum_val + sorted_times(i)
    end do
    stats%mean_time = sum_val / real(n, real64)
    
    ! Median
    if (mod(n, 2) == 0) then
      stats%median_time = (sorted_times(n/2) + sorted_times(n/2 + 1)) / 2.0_real64
    else
      stats%median_time = sorted_times((n+1)/2)
    end if
    
    ! Standard deviation
    sum_sq = 0.0_real64
    do i = 1, n
      sum_sq = sum_sq + (sorted_times(i) - stats%mean_time)**2
    end do
    if (n > 1) then
      stats%stddev_time = sqrt(sum_sq / real(n-1, real64))
    else
      stats%stddev_time = 0.0_real64
    end if
    
    deallocate(sorted_times)
  end subroutine compute_statistics
  
  recursive subroutine quicksort(a, lo, hi)
    real(real64), intent(inout) :: a(:)
    integer(int32), intent(in) :: lo, hi
    integer(int32) :: p
    
    if (lo < hi) then
      p = partition(a, lo, hi)
      call quicksort(a, lo, p-1_int32)
      call quicksort(a, p+1_int32, hi)
    end if
  end subroutine quicksort
  
  integer(int32) function partition(a, lo, hi)
    real(real64), intent(inout) :: a(:)
    integer(int32), intent(in) :: lo, hi
    real(real64) :: pivot, temp
    integer(int32) :: i, j
    
    pivot = a(hi)
    i = lo - 1_int32
    
    do j = lo, hi - 1_int32
      if (a(j) <= pivot) then
        i = i + 1_int32
        temp = a(i)
        a(i) = a(j)
        a(j) = temp
      end if
    end do
    
    temp = a(i+1_int32)
    a(i+1_int32) = a(hi)
    a(hi) = temp
    
    partition = i + 1_int32
  end function partition
  
end module timing_mod

module memory_mod
  use iso_fortran_env, only: real64, int64
  use constants_mod
  implicit none
  
contains

  subroutine allocate_aligned(ptr, n, align)
    real(real64), allocatable, intent(out) :: ptr(:)
    integer(int64), intent(in) :: n
    integer, intent(in) :: align
    
    ! Standard allocation (compiler may align automatically)
    allocate(ptr(n))
    
  end subroutine allocate_aligned
  
  subroutine init_arrays_numa(A, B, C, D, E, matrix_A, N, M)
    real(real64), intent(out) :: A(:), B(:), C(:), D(:), E(:)
    real(real64), intent(out) :: matrix_A(:,:)
    integer(int64), intent(in) :: N, M
    integer(int64) :: i, j
    
    ! NUMA-aware initialization with first-touch policy
    !$omp parallel do schedule(static) private(i)
    do i = 1, N
      A(i) = 0.0_real64
      B(i) = 1.0_real64 + 1.0e-6_real64 * real(mod(i, 1000_int64), real64)
      C(i) = 2.0_real64 - 1.0e-6_real64 * real(mod(i, 1000_int64), real64)
      D(i) = 0.5_real64
      E(i) = 1.5_real64
    end do
    !$omp end parallel do
    
    ! Initialize matrix for DGEMV
    !$omp parallel do collapse(2) schedule(static) private(i,j)
    do j = 1, M
      do i = 1, M
        matrix_A(i,j) = 1.0_real64 / real(i + j, real64)
      end do
    end do
    !$omp end parallel do
  end subroutine init_arrays_numa
  
end module memory_mod

module kernels_mod
  use iso_fortran_env, only: real64, int64
  use constants_mod
  implicit none
  
contains

  ! AXPY: Y = a*X + Y (2 FLOP/elem, 2 loads + 1 store)
  subroutine ker_axpy(N, Y, X, a)
    integer(int64), intent(in) :: N
    real(real64), intent(inout) :: Y(:)
    real(real64), intent(in) :: X(:), a
    integer(int64) :: i
    
    !$omp parallel do simd schedule(static) if(N>100000)
    do i = 1, N
      Y(i) = a * X(i) + Y(i)
    end do
    !$omp end parallel do simd
  end subroutine ker_axpy
  
  ! TRIAD: A = B + s*C (2 FLOP/elem, 2 loads + 1 store)
  subroutine ker_triad(N, A, B, C, s)
    integer(int64), intent(in) :: N
    real(real64), intent(out) :: A(:)
    real(real64), intent(in) :: B(:), C(:), s
    integer(int64) :: i
    
    !$omp parallel do simd schedule(static) if(N>100000)
    do i = 1, N
      A(i) = B(i) + s * C(i)
    end do
    !$omp end parallel do simd
  end subroutine ker_triad
  
  ! UPDATE3: A = alpha*B + beta*C + D (4 FLOP/elem, 3 loads + 1 store)
  subroutine ker_update3(N, A, B, C, D, alpha, beta)
    integer(int64), intent(in) :: N
    real(real64), intent(out) :: A(:)
    real(real64), intent(in) :: B(:), C(:), D(:), alpha, beta
    integer(int64) :: i
    
    !$omp parallel do simd schedule(static) if(N>100000)
    do i = 1, N
      A(i) = alpha * B(i) + beta * C(i) + D(i)
    end do
    !$omp end parallel do simd
  end subroutine ker_update3
  
  ! DOT: result = sum(X(i)*Y(i)) (2 FLOP/elem, 2 loads)
  subroutine ker_dot(N, result, X, Y)
    integer(int64), intent(in) :: N
    real(real64), intent(out) :: result
    real(real64), intent(in) :: X(:), Y(:)
    integer(int64) :: i
    
    result = 0.0_real64
    !$omp parallel do simd reduction(+:result) schedule(static) if(N>100000)
    do i = 1, N
      result = result + X(i) * Y(i)
    end do
    !$omp end parallel do simd
  end subroutine ker_dot
  
  ! DGEMV: Y = alpha*A*X + beta*Y (2*M*M FLOP, M*M + M loads + M store)
  subroutine ker_dgemv(M, Y, A, X, alpha, beta)
    integer(int64), intent(in) :: M
    real(real64), intent(inout) :: Y(:)
    real(real64), intent(in) :: A(:,:), X(:)
    real(real64), intent(in) :: alpha, beta
    integer(int64) :: i, j
    real(real64) :: temp
    
    !$omp parallel do private(i,j,temp) schedule(static) if(M>1000)
    do i = 1, M
      temp = 0.0_real64
      !$omp simd reduction(+:temp)
      do j = 1, M
        temp = temp + A(i,j) * X(j)
      end do
      Y(i) = alpha * temp + beta * Y(i)
    end do
    !$omp end parallel do
  end subroutine ker_dgemv
  
  ! 5-point stencil: A(i) = B(i-2) + B(i-1) + B(i) + B(i+1) + B(i+2)
  subroutine ker_stencil5(N, A, B)
    integer(int64), intent(in) :: N
    real(real64), intent(out) :: A(:)
    real(real64), intent(in) :: B(:)
    integer(int64) :: i
    
    ! Handle boundaries
    A(1) = B(1)
    A(2) = B(1) + B(2) + B(3)
    
    !$omp parallel do simd schedule(static) if(N>100000)
    do i = 3, N-2
      A(i) = B(i-2) + B(i-1) + B(i) + B(i+1) + B(i+2)
    end do
    !$omp end parallel do simd
    
    A(N-1) = B(N-2) + B(N-1) + B(N)
    A(N) = B(N)
  end subroutine ker_stencil5
  
end module kernels_mod

module validation_mod
  use iso_fortran_env, only: real64, int64
  use constants_mod
  implicit none
  
contains

  subroutine validate_axpy(N, Y, X, a, passed)
    integer(int64), intent(in) :: N
    real(real64), intent(in) :: Y(:), X(:), a
    logical, intent(out) :: passed
    real(real64), allocatable :: Y_ref(:)
    real(real64) :: max_err
    integer(int64) :: i
    
    allocate(Y_ref(N))
    
    ! Initialize reference
    do i = 1, N
      Y_ref(i) = 0.0_real64
    end do
    
    ! Compute reference solution
    do i = 1, N
      Y_ref(i) = a * X(i) + Y_ref(i)
    end do
    
    ! Check error
    max_err = 0.0_real64
    do i = 1, N
      max_err = max(max_err, abs(Y(i) - Y_ref(i)))
    end do
    
    passed = (max_err < VALIDATION_TOL)
    
    if (.not. passed) then
      print *, 'AXPY validation failed: max_err = ', max_err
    end if
    
    deallocate(Y_ref)
  end subroutine validate_axpy
  
end module validation_mod

module output_mod
  use iso_fortran_env, only: real64, int64
  use timing_mod
  use omp_lib
  implicit none
  
contains

  subroutine write_json_output(filename, kernel_name, N, stats, ai, flops_per_elem, bytes_per_elem)
    character(len=*), intent(in) :: filename, kernel_name
    integer(int64), intent(in) :: N
    type(perf_stats), intent(in) :: stats
    real(real64), intent(in) :: ai, flops_per_elem, bytes_per_elem
    integer :: unit
    
    open(newunit=unit, file=filename, status='replace', action='write')
    
    write(unit, '(A)') '{'
    write(unit, '(A,A,A)') '  "kernel": "', trim(kernel_name), '",'
    write(unit, '(A,I0,A)') '  "N": ', N, ','
    write(unit, '(A,I0,A)') '  "num_runs": ', stats%num_runs, ','
    write(unit, '(A,F12.6,A)') '  "min_time_sec": ', stats%min_time, ','
    write(unit, '(A,F12.6,A)') '  "median_time_sec": ', stats%median_time, ','
    write(unit, '(A,F12.6,A)') '  "mean_time_sec": ', stats%mean_time, ','
    write(unit, '(A,F12.6,A)') '  "stddev_time_sec": ', stats%stddev_time, ','
    write(unit, '(A,F12.3,A)') '  "gflops_peak": ', stats%gflops_peak, ','
    write(unit, '(A,F12.3,A)') '  "gflops_mean": ', stats%gflops_mean, ','
    write(unit, '(A,F12.3,A)') '  "bandwidth_GB_s_peak": ', stats%bandwidth_peak, ','
    write(unit, '(A,F12.3,A)') '  "bandwidth_GB_s_mean": ', stats%bandwidth_mean, ','
    write(unit, '(A,F12.6,A)') '  "arithmetic_intensity": ', ai, ','
    write(unit, '(A,F6.1,A)') '  "flops_per_element": ', flops_per_elem, ','
    write(unit, '(A,F6.1,A)') '  "bytes_per_element": ', bytes_per_elem, ','
    write(unit, '(A,I0)') '  "num_threads": ', omp_get_max_threads()
    write(unit, '(A)') '}'
    
    close(unit)
  end subroutine write_json_output
  
  subroutine write_csv_header(unit)
    integer, intent(in) :: unit
    
    write(unit, '(A)') 'kernel,N,threads,min_time,median_time,mean_time,stddev_time,' // &
                      'gflops_peak,gflops_mean,bandwidth_peak_GB_s,bandwidth_mean_GB_s,' // &
                      'arithmetic_intensity,flops_per_elem,bytes_per_elem'
  end subroutine write_csv_header
  
  subroutine write_csv_row(unit, kernel_name, N, stats, ai, flops_per_elem, bytes_per_elem)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: kernel_name
    integer(int64), intent(in) :: N
    type(perf_stats), intent(in) :: stats
    real(real64), intent(in) :: ai, flops_per_elem, bytes_per_elem
    
    write(unit, '(A,",",I0,",",I0,",",F12.6,",",F12.6,",",F12.6,",",F12.6,",",' // &
                'F12.3,",",F12.3,",",F12.3,",",F12.3,",",F12.6,",",F6.1,",",F6.1)') &
          trim(kernel_name), N, omp_get_max_threads(), &
          stats%min_time, stats%median_time, stats%mean_time, stats%stddev_time, &
          stats%gflops_peak, stats%gflops_mean, &
          stats%bandwidth_peak, stats%bandwidth_mean, &
          ai, flops_per_elem, bytes_per_elem
  end subroutine write_csv_row
  
end module output_mod

module benchmark_mod
  use iso_fortran_env, only: real64, int64
  use constants_mod
  use timing_mod
  use kernels_mod
  implicit none
  
  ! Cache model selection
  character(len=32) :: global_cache_model = 'realistic'
  
contains

  subroutine set_cache_model(model)
    character(len=*), intent(in) :: model
    global_cache_model = model
  end subroutine set_cache_model

  subroutine run_kernel_benchmark(kernel_id, N, M, A, B, C, D, E, matrix_A, &
                                 alpha, beta, iters, warmup, stats_runs, &
                                 stats, ai, flops_per_elem, bytes_per_elem)
    integer, intent(in) :: kernel_id
    integer(int64), intent(in) :: N, M, iters, warmup, stats_runs
    real(real64), intent(inout) :: A(:), B(:), C(:), D(:), E(:)
    real(real64), intent(inout) :: matrix_A(:,:)
    real(real64), intent(in) :: alpha, beta
    type(perf_stats), intent(out) :: stats
    real(real64), intent(out) :: ai, flops_per_elem, bytes_per_elem
    
    real(real64), allocatable :: times(:)
    real(real64) :: t0, t1, dot_result
    real(real64) :: flops_total, bytes_total
    integer :: run, k
    integer(int64) :: work_size
    
    allocate(times(stats_runs))
    
    ! Determine work size based on kernel
    work_size = N
    if (kernel_id == KERNEL_DGEMV) work_size = M
    
    ! Get FLOP and byte counts
    call get_kernel_metrics_cache_aware(kernel_id, work_size, flops_per_elem, bytes_per_elem)
    ai = flops_per_elem / bytes_per_elem
    
    ! Warmup
    do k = 1, warmup
      call execute_kernel(kernel_id, N, M, A, B, C, D, E, matrix_A, &
                         alpha, beta, dot_result)
    end do
    
    ! Timing runs
    do run = 1, stats_runs
      call fence_io()
      
      t0 = walltime()
      do k = 1, iters
        call execute_kernel(kernel_id, N, M, A, B, C, D, E, matrix_A, &
                           alpha, beta, dot_result)
      end do
      t1 = walltime()
      
      times(run) = t1 - t0
    end do
    
    ! Compute statistics
    call compute_statistics(times, int(stats_runs), stats)
    
    ! Calculate performance metrics
    flops_total = real(work_size, real64) * real(iters, real64) * flops_per_elem / 1.0e9_real64
    bytes_total = real(work_size, real64) * real(iters, real64) * bytes_per_elem / 1.0e9_real64
    
    stats%gflops_peak = flops_total / stats%min_time
    stats%gflops_mean = flops_total / stats%mean_time
    stats%bandwidth_peak = bytes_total / stats%min_time
    stats%bandwidth_mean = bytes_total / stats%mean_time
    
    deallocate(times)
  end subroutine run_kernel_benchmark
  
  subroutine execute_kernel(kernel_id, N, M, A, B, C, D, E, matrix_A, &
                           alpha, beta, dot_result)
    integer, intent(in) :: kernel_id
    integer(int64), intent(in) :: N, M
    real(real64), intent(inout) :: A(:), B(:), C(:), D(:), E(:)
    real(real64), intent(inout) :: matrix_A(:,:)
    real(real64), intent(in) :: alpha, beta
    real(real64), intent(out) :: dot_result
    
    select case(kernel_id)
    case(KERNEL_AXPY)
      call ker_axpy(N, A, B, alpha)
    case(KERNEL_TRIAD)
      call ker_triad(N, A, B, C, alpha)
    case(KERNEL_UPDATE3)
      call ker_update3(N, A, B, C, D, alpha, beta)
    case(KERNEL_DOT)
      call ker_dot(N, dot_result, B, C)
    case(KERNEL_DGEMV)
      call ker_dgemv(M, E(1:M), matrix_A, B(1:M), alpha, beta)
    case(KERNEL_STENCIL5)
      call ker_stencil5(N, A, B)
    end select
  end subroutine execute_kernel
  
  subroutine get_kernel_metrics_cache_aware(kernel_id, work_size, flops_per_elem, bytes_per_elem)
    integer, intent(in) :: kernel_id
    integer(int64), intent(in) :: work_size
    real(real64), intent(out) :: flops_per_elem, bytes_per_elem
    
    select case(kernel_id)
    case(KERNEL_AXPY)
      flops_per_elem = 2.0_real64  ! mul + add
      bytes_per_elem = 24.0_real64  ! X load + Y load/store
      
    case(KERNEL_TRIAD)
      flops_per_elem = 2.0_real64
      bytes_per_elem = 24.0_real64  ! B load + C load + A store
      
    case(KERNEL_UPDATE3)
      flops_per_elem = 4.0_real64
      bytes_per_elem = 32.0_real64  ! B,C,D loads + A store
      
    case(KERNEL_DOT)
      flops_per_elem = 2.0_real64
      bytes_per_elem = 16.0_real64  ! X,Y loads only
      
    case(KERNEL_DGEMV)
      flops_per_elem = 2.0_real64 * real(work_size, real64)
      bytes_per_elem = 8.0_real64 * (real(work_size, real64) + 2.0_real64)
      
    case(KERNEL_STENCIL5)
      flops_per_elem = 4.0_real64  ! 4 adds
      
      if (global_cache_model == 'naive') then
        ! Naive model: count all accesses
        bytes_per_elem = 48.0_real64  ! 5 loads + 1 store
      else  ! 'realistic' model
        ! Cache-aware model: consider spatial locality
        ! In streaming access pattern, each cache line is loaded once
        ! Assuming 64-byte cache lines (8 elements), the 5-point stencil
        ! effectively loads each element ~1.5 times due to overlap
        bytes_per_elem = 20.0_real64  ! ~1.5 loads + 1 store
      end if
      
    end select
  end subroutine get_kernel_metrics_cache_aware
  
  subroutine get_kernel_metrics(kernel_id, work_size, flops_per_elem, bytes_per_elem)
    ! Wrapper for backward compatibility
    integer, intent(in) :: kernel_id
    integer(int64), intent(in) :: work_size  
    real(real64), intent(out) :: flops_per_elem, bytes_per_elem
    
    call get_kernel_metrics_cache_aware(kernel_id, work_size, flops_per_elem, bytes_per_elem)
  end subroutine get_kernel_metrics
  
  character(len=20) function get_kernel_name(kernel_id)
    integer, intent(in) :: kernel_id
    
    select case(kernel_id)
    case(KERNEL_AXPY)
      get_kernel_name = 'AXPY'
    case(KERNEL_TRIAD)
      get_kernel_name = 'TRIAD'
    case(KERNEL_UPDATE3)
      get_kernel_name = 'UPDATE3'
    case(KERNEL_DOT)
      get_kernel_name = 'DOT'
    case(KERNEL_DGEMV)
      get_kernel_name = 'DGEMV'
    case(KERNEL_STENCIL5)
      get_kernel_name = 'STENCIL5'
    case default
      get_kernel_name = 'UNKNOWN'
    end select
  end function get_kernel_name
  
end module benchmark_mod

! Main program moved outside of module for NVFORTRAN compatibility
program roofline_performance_suite
  use iso_fortran_env, only: real64, int64
  use constants_mod
  use timing_mod
  use memory_mod
  use kernels_mod
  use validation_mod
  use output_mod
  use benchmark_mod
  use omp_lib
  implicit none
  
  ! Parameters
  integer(int64) :: N, M, iters, warmup, stats_runs
  integer :: kernel_choice
  character(len=32) :: output_format
  character(len=256) :: output_file
  character(len=32) :: cache_model
  logical :: validate_flag
  
  ! Arrays
  real(real64), allocatable :: A(:), B(:), C(:), D(:), E(:)
  real(real64), allocatable :: matrix_A(:,:)
  
  ! Benchmark parameters
  real(real64) :: alpha, beta
  
  ! Performance metrics
  type(perf_stats) :: stats
  real(real64) :: ai, flops_per_elem, bytes_per_elem
  
  ! Other variables
  integer :: csv_unit
  integer :: i, kernel_id
  integer, allocatable :: kernel_list(:)
  character(len=20) :: kernel_name
  logical :: passed
  
  ! Parse command line arguments
  call parse_args(N, M, iters, warmup, stats_runs, kernel_choice, &
                  output_format, output_file, cache_model, validate_flag, alpha, beta)
  
  ! Set cache model for benchmarks
  call set_cache_model(cache_model)
  
  ! Allocate arrays with alignment
  call allocate_aligned(A, N, ALIGNMENT)
  call allocate_aligned(B, N, ALIGNMENT)
  call allocate_aligned(C, N, ALIGNMENT)
  call allocate_aligned(D, N, ALIGNMENT)
  call allocate_aligned(E, N, ALIGNMENT)
  allocate(matrix_A(M, M))
  
  ! Initialize arrays with NUMA awareness
  call init_arrays_numa(A, B, C, D, E, matrix_A, N, M)
  
  ! Determine which kernels to run
  if (kernel_choice == KERNEL_ALL) then
    allocate(kernel_list(6))
    kernel_list = [KERNEL_AXPY, KERNEL_TRIAD, KERNEL_UPDATE3, &
                   KERNEL_DOT, KERNEL_DGEMV, KERNEL_STENCIL5]
  else
    allocate(kernel_list(1))
    kernel_list(1) = kernel_choice
  end if
  
  ! Open CSV file if needed
  if (output_format == 'csv') then
    open(newunit=csv_unit, file=trim(output_file), status='replace')
    call write_csv_header(csv_unit)
  end if
  
  ! Print header
  call print_header(N, M, iters, warmup, stats_runs, cache_model)
  
  ! Run benchmarks
  do i = 1, size(kernel_list)
    kernel_id = kernel_list(i)
    kernel_name = get_kernel_name(kernel_id)
    
    print *, '------------------------------------------------------------'
    print *, 'Running kernel: ', trim(kernel_name)
    
    ! Run benchmark
    call run_kernel_benchmark(kernel_id, N, M, A, B, C, D, E, matrix_A, &
                             alpha, beta, iters, warmup, stats_runs, &
                             stats, ai, flops_per_elem, bytes_per_elem)
    
    ! Print results
    call print_results(kernel_name, N, stats, ai, flops_per_elem, bytes_per_elem)
    
    ! Write output
    if (output_format == 'json') then
      write(output_file, '(A,A,A)') trim(kernel_name), '_results.json'
      call write_json_output(output_file, kernel_name, N, stats, ai, &
                            flops_per_elem, bytes_per_elem)
    else if (output_format == 'csv') then
      call write_csv_row(csv_unit, kernel_name, N, stats, ai, &
                        flops_per_elem, bytes_per_elem)
    end if
    
    ! Validate if requested
    if (validate_flag .and. kernel_id == KERNEL_AXPY) then
      call validate_axpy(N, A, B, alpha, passed)
      if (passed) print *, 'Validation: PASSED'
    end if
  end do
  
  ! Close CSV file
  if (output_format == 'csv') close(csv_unit)
  
  ! Print footer
  print *, '------------------------------------------------------------'
  print *, 'Benchmark completed successfully!'
  
  ! Cleanup
  deallocate(A, B, C, D, E, matrix_A, kernel_list)
  
contains

  subroutine parse_args(N, M, iters, warmup, stats_runs, kernel_choice, &
                       output_format, output_file, cache_model, validate_flag, alpha, beta)
    integer(int64), intent(out) :: N, M, iters, warmup, stats_runs
    integer, intent(out) :: kernel_choice
    character(len=*), intent(out) :: output_format, output_file, cache_model
    logical, intent(out) :: validate_flag
    real(real64), intent(out) :: alpha, beta
    
    character(len=256) :: arg
    integer :: i, ierr
    
    ! Defaults
    N = 100000000_int64
    M = 1000_int64
    iters = 5_int64
    warmup = 2_int64
    stats_runs = 5
    kernel_choice = KERNEL_ALL
    output_format = 'text'
    output_file = 'roofline_results.csv'
    cache_model = 'realistic'  ! 'naive' or 'realistic'
    validate_flag = .false.
    alpha = 2.0_real64
    beta = 3.0_real64
    
    do i = 1, command_argument_count()
      call get_command_argument(i, arg)
      
      if (index(arg, '--N=') == 1) then
        read(arg(5:), *, iostat=ierr) N
      else if (index(arg, '--M=') == 1) then
        read(arg(5:), *, iostat=ierr) M
      else if (index(arg, '--iters=') == 1) then
        read(arg(9:), *, iostat=ierr) iters
      else if (index(arg, '--warmup=') == 1) then
        read(arg(10:), *, iostat=ierr) warmup
      else if (index(arg, '--stats=') == 1) then
        read(arg(9:), *, iostat=ierr) stats_runs
      else if (index(arg, '--kernel=') == 1) then
        arg = arg(10:)
        if (trim(arg) == 'axpy') kernel_choice = KERNEL_AXPY
        if (trim(arg) == 'triad') kernel_choice = KERNEL_TRIAD
        if (trim(arg) == 'update3') kernel_choice = KERNEL_UPDATE3
        if (trim(arg) == 'dot') kernel_choice = KERNEL_DOT
        if (trim(arg) == 'dgemv') kernel_choice = KERNEL_DGEMV
        if (trim(arg) == 'stencil5') kernel_choice = KERNEL_STENCIL5
        if (trim(arg) == 'all') kernel_choice = KERNEL_ALL
      else if (index(arg, '--output=') == 1) then
        output_format = trim(adjustl(arg(10:)))
      else if (index(arg, '--outfile=') == 1) then
        output_file = trim(adjustl(arg(11:)))
      else if (index(arg, '--cache=') == 1) then
        cache_model = trim(adjustl(arg(9:)))
      else if (trim(arg) == '--validate') then
        validate_flag = .true.
      else if (trim(arg) == '--help' .or. trim(arg) == '-h') then
        call print_usage()
        stop
      end if
    end do
  end subroutine parse_args
  
  subroutine print_usage()
    print *, 'Roofline Performance Analysis Suite'
    print *, 'Usage: ./roofline_suite [options]'
    print *, ''
    print *, 'Options:'
    print *, '  --N=<size>       Vector size (default: 100000000)'
    print *, '  --M=<size>       Matrix dimension for DGEMV (default: 1000)'
    print *, '  --iters=<n>      Number of iterations (default: 5)'
    print *, '  --warmup=<n>     Warmup iterations (default: 2)'
    print *, '  --stats=<n>      Number of runs for statistics (default: 5)'
    print *, '  --kernel=<name>  Kernel: axpy|triad|update3|dot|dgemv|stencil5|all (default: all)'
    print *, '  --output=<fmt>   Output format: text|json|csv (default: text)'
    print *, '  --outfile=<file> Output filename (default: roofline_results.csv)'
    print *, '  --validate       Enable validation'
    print *, '  --help           Show this help'
  end subroutine print_usage
  
  subroutine print_header(N, M, iters, warmup, stats_runs, cache_model)
    integer(int64), intent(in) :: N, M, iters, warmup, stats_runs
    character(len=*), intent(in) :: cache_model
    
    print *, '============================================================'
    print *, 'ROOFLINE PERFORMANCE ANALYSIS SUITE'
    print *, '============================================================'
    print '(A,I0)', ' Vector size (N)     : ', N
    print '(A,I0)', ' Matrix size (M)     : ', M
    print '(A,I0)', ' Iterations          : ', iters
    print '(A,I0)', ' Warmup iterations   : ', warmup
    print '(A,I0)', ' Statistical runs    : ', stats_runs
    print '(A,I0)', ' OpenMP threads      : ', omp_get_max_threads()
    print '(A,A)',  ' Cache model         : ', trim(cache_model)
    print *, '============================================================'
  end subroutine print_header
  
  subroutine print_results(kernel_name, N, stats, ai, flops_per_elem, bytes_per_elem)
    character(len=*), intent(in) :: kernel_name
    integer(int64), intent(in) :: N
    type(perf_stats), intent(in) :: stats
    real(real64), intent(in) :: ai, flops_per_elem, bytes_per_elem
    
    print '(A,F12.6,A)', ' Min time        : ', stats%min_time, ' s'
    print '(A,F12.6,A)', ' Median time     : ', stats%median_time, ' s'
    print '(A,F12.6,A)', ' Mean time       : ', stats%mean_time, ' s'
    print '(A,F12.6,A)', ' Std dev         : ', stats%stddev_time, ' s'
    print '(A,F12.3)',   ' Peak GFLOPS/s   : ', stats%gflops_peak
    print '(A,F12.3)',   ' Mean GFLOPS/s   : ', stats%gflops_mean
    print '(A,F12.3)',   ' Peak BW (GB/s)  : ', stats%bandwidth_peak
    print '(A,F12.3)',   ' Mean BW (GB/s)  : ', stats%bandwidth_mean
    print '(A,F12.6)',   ' AI (FLOP/Byte)  : ', ai
  end subroutine print_results
  
end program roofline_performance_suite
