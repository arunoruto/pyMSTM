!
!  mstm_f2py.f90 -- f2py-facing wrapper for Python interop (replaces the
!  iso_c_binding/ctypes wrapper mstm_wrapper.f90 during the migration).
!
!  Compile with mpidefs-serial.f90 (no MPI needed for single-process).
!
!  Many "pure setter/getter" module variables (mie_epsilon, t_matrix_order,
!  azimuthal_average, etc.) are NOT wrapped here at all -- f2py exposes
!  public module variables directly as Python attributes on the compiled
!  extension's per-module namespace (e.g. ext.inputinterface.mie_epsilon,
!  ext.spheredata.t_matrix_order). Only subroutines with real orchestration
!  logic (allocation, defaults fan-out, multi-step calculations) live here.
!

module mstm_f2py_bindings
use inputinterface
use solver
use spheredata
use scatprops
use translation
use mpidefs
use numconstants
use surface_subroutines
use periodic_lattice_subroutines
use random_sphere_configuration
implicit none

contains

!
!  Initialize the MSTM subsystem (MPI init + set safe defaults).
!
subroutine mstm_init()
    implicit none
    call mstm_mpi(mpi_command='init')
    call mstm_mpi(mpi_command='rank', mpi_rank=global_rank)

    ! serial: no MPI ranks beyond 0
    run_print_unit = 6
    ! suppress most print output
    print_intermediate_results = 0
    light_up = .false.

    ! Defaults for a pure cluster (no plane boundaries, no periodic lattice)
    number_plane_boundaries = 0
    plane_surface_present = .false.
    periodic_lattice = .false.
    random_configuration = .false.
    configuration_average = .false.
    random_orientation = .false.
    incidence_average = .false.
    calculate_near_field = .false.
    calculate_up_down_scattering = .false.
    calculate_scattering_matrix = .true.
    single_origin_expansion = .true.
    incident_frame = .true.
    azimuthal_average = .false.
    numerical_azimuthal_average = .false.
    normalize_s11 = .true.
    print_sphere_data = .false.
    print_timings = .false.
    print_random_configuration = .false.
    copy_input_file = .false.
    append_output_file = .false.
    check_positions = .false.
    reflection_model = .false.
    input_effective_medium_simulation = .false.
    effective_medium_simulation = .false.
    store_translation_matrix = .false.
    store_surface_matrix = .true.
    normalize_solution_error = .true.
    fft_translation_option = .false.
    input_fft_translation_option = .false.

    ! Scale factors
    length_scale_factor = 1.d0
    ref_index_scale_factor = (1.d0, 0.d0)

    ! Solver defaults
    solution_method = 'i'
    solution_epsilon = 1.d-6
    max_iterations = 10000
    t_matrix_convergence_epsilon = 1.d-6
    mie_epsilon = 1.d-6
    translation_epsilon = 1.d-5

    ! Incident field defaults
    incident_alpha_deg = 0.d0
    incident_beta_deg = 0.d0
    incident_sin_beta = 0.d0
    incident_direction = 1
    incident_beta_specified = .false.
    gaussian_beam_constant = 0.d0

    ! Scattering matrix defaults
    scattering_map_model = 0
    scattering_map_dimension = 15
    scat_mat_amin = 0.d0
    scat_mat_amax = 180.d0
    scat_mat_ldim = -scattering_map_dimension
    scat_mat_udim = scattering_map_dimension
    scat_mat_mdim = 16 + 16  ! up + down, 16 elements each

    ! Other
    n_nest_loops = 0
    repeat_run = .false.
    first_run = .true.

    ! Host sphere defaults
    host_sphere_ref_index = (1.d0, 0.d0)
    random_configuration_host = .false.
    number_spheres_specified = .true.

    ! Medium ref index
    medium_ref_index = (1.d0, 0.d0)
    medium_ref_index_specified = .false.

    ! Layer defaults (normally set via DATA statements in mstm-modules-33.f90,
    ! but those use a named-parameter repeat count in "N*value" DATA syntax
    ! that crashes numpy.f2py's crackfortran parser -- see
    ! src/pymstm/_fortran/patches/mstm-modules-33-f2py-data-fix.patch, which
    ! removes them from the source used for the f2py build. Replicated here
    ! so behavior is identical to the unpatched source.)
    layer_thickness = 1.d0
    layer_ref_index(0) = (1.d0, 0.d0)
    layer_ref_index(1:max_number_plane_boundaries) = (1.d0, 0.d0)

    ! Same reason: a DATA statement mixing effective_medium_simulation
    ! (logical) and effective_ref_index (complex(8), not first in the list)
    ! also crashes crackfortran. effective_medium_simulation is already set
    ! to .false. above; effective_ref_index's default is replicated here.
    effective_ref_index = (1.d0, 0.d0)

end subroutine mstm_init


!
!  Set the number of spheres and allocate sphere data arrays.
!    n          -- number of spheres
!    orders     -- per-sphere Mie order (int, length n)
!    radii      -- per-sphere radius (double, length n)
!    pos        -- positions, shape (3, n): pos(1,i),pos(2,i),pos(3,i) = x,y,z
!    ref_re/ref_im -- refractive index real/imag parts per sphere (length n)
!
subroutine mstm_set_spheres(n, orders, radii, pos, ref_re, ref_im)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: orders(n)
    real(8), intent(in) :: radii(n), pos(3, n)
    real(8), intent(in) :: ref_re(n), ref_im(n)
    integer :: i

    ! Deallocate if previously allocated
    if (allocated(sphere_order)) then
        deallocate(sphere_order, sphere_radius, sphere_position, &
            sphere_ref_index, host_sphere, number_field_expansions, &
            sphere_excitation_switch, sphere_index, optically_active, &
            sphere_block, sphere_offset, mie_offset, sphere_layer, &
            sphere_depth, qext_mie, qabs_mie)
    endif
    if (allocated(sphere_links)) deallocate(sphere_links)

    ! Set number of spheres
    number_spheres = n
    input_number_spheres = n
    number_host_spheres = 0

    ! Allocate per-sphere arrays
    allocate(sphere_order(n), sphere_radius(n), sphere_position(3,n), &
        sphere_ref_index(2,0:n), host_sphere(n), &
        number_field_expansions(n), sphere_excitation_switch(n), &
        sphere_index(n))
    allocate(optically_active(n), sphere_block(n), sphere_offset(n+1), &
        mie_offset(n+1), qext_mie(n), qabs_mie(n))

    do i = 1, n
        sphere_order(i) = orders(i)
        sphere_radius(i) = radii(i) * length_scale_factor
        sphere_position(1,i) = pos(1,i) * length_scale_factor
        sphere_position(2,i) = pos(2,i) * length_scale_factor
        sphere_position(3,i) = pos(3,i) * length_scale_factor
        sphere_ref_index(1,i) = cmplx(ref_re(i), ref_im(i), 8) &
            * ref_index_scale_factor
        sphere_ref_index(2,i) = sphere_ref_index(1,i)  ! same for both polarizations
        host_sphere(i) = 0
        sphere_index(i) = i
    enddo
    ! host sphere (0) = medium
    sphere_ref_index(:,0) = medium_ref_index

    ! default: excite all spheres
    sphere_excitation_switch = .true.
    number_excited_spheres = n

end subroutine mstm_set_spheres


!
!  Set medium (host) refractive index.
!
subroutine mstm_set_medium_ref(re, im)
    implicit none
    real(8), intent(in) :: re, im
    medium_ref_index = cmplx(re, im, 8)
    medium_ref_index_specified = .true.
    layer_ref_index(0) = medium_ref_index
    if (allocated(sphere_ref_index)) then
        sphere_ref_index(:,0) = medium_ref_index
    endif
end subroutine mstm_set_medium_ref


!
!  Set incident field parameters (angles in degrees).
!
!  NOTE: incident_sin_beta and incident_direction are NOT computed here.
!  The authoritative CLI logic (mstm-input-37.f90 ~1310-1318) derives them
!  from beta_deg via Snell's law, dividing by the real part of the
!  boundary-layer refractive index at the incidence side -- which requires
!  layer_ref_index/number_plane_boundaries to already be finalized (i.e.
!  after any set_layers() call, not necessarily before it, since callers
!  may configure layers before or after incidence). That derivation is done
!  in mstm_prepare() instead, once all inputs are known. The old
!  ctypes-era mstm_set_incident_c computed a naive sin(beta_deg) here and
!  took `direction` as a caller-supplied literal, ignoring the medium's
!  refractive index and never re-deriving direction from beta_deg -- a
!  latent bug found during this migration (harmless only because the
!  medium defaults to a real refractive index of 1 and no caller combines
!  oblique incidence with a layered medium).
!
subroutine mstm_set_incident(alpha_deg, beta_deg)
    implicit none
    real(8), intent(in) :: alpha_deg, beta_deg
    incident_alpha_deg = alpha_deg
    incident_beta_deg = beta_deg
    if (beta_deg == 0.d0) then
        incident_beta_specified = .false.
    else
        incident_beta_specified = .true.
    endif
end subroutine mstm_set_incident


!
!  Set solver parameters.
!  method: 0 = iterative BiCG, 1 = direct LU
!
subroutine mstm_set_solver_params(eps, maxiter, method)
    implicit none
    real(8), intent(in) :: eps
    integer, intent(in) :: maxiter, method
    solution_epsilon = eps
    max_iterations = maxiter
    if (method == 1) then
        solution_method = 'd'
    else
        solution_method = 'i'
    endif
end subroutine mstm_set_solver_params


!
!  Set the number of plane boundary layers and initialize defaults.
!
subroutine mstm_set_layer_count(n_layers)
    implicit none
    integer, intent(in) :: n_layers
    integer :: i
    number_plane_boundaries = n_layers
    plane_surface_present = (n_layers > 0)
    if (n_layers > 0) then
        ! Initialize layers to reasonable defaults
        do i = 1, min(n_layers, max_number_plane_boundaries)
            layer_thickness(i) = 1.d0
            layer_ref_index(i) = (1.d0, 0.d0)
        enddo
        ! Ensure allocation
        if (allocated(plane_boundary_position)) &
            deallocate(plane_boundary_position)
        allocate(plane_boundary_position(max(n_layers, 1)))
    endif
end subroutine mstm_set_layer_count


!
!  Set thickness of a specific layer (1-indexed, 1 = first layer).
!
subroutine mstm_set_layer_thickness(layer, thickness)
    implicit none
    integer, intent(in) :: layer
    real(8), intent(in) :: thickness
    if (layer >= 1 .and. layer <= max_number_plane_boundaries) then
        layer_thickness(layer) = thickness
    endif
end subroutine mstm_set_layer_thickness


!
!  Set refractive index of a specific layer (1-indexed, 0 = incident medium).
!
subroutine mstm_set_layer_ref_index(layer, re, im)
    implicit none
    integer, intent(in) :: layer
    real(8), intent(in) :: re, im
    if (layer >= 0 .and. layer <= max_number_plane_boundaries) then
        layer_ref_index(layer) = cmplx(re, im, 8)
        if (layer == 0) then
            medium_ref_index = layer_ref_index(0)
            medium_ref_index_specified = .true.
        endif
    endif
end subroutine mstm_set_layer_ref_index


!
!  Set periodic lattice configuration.
!  cell_wx, cell_wy = x and y dimensions of the unit cell
!  phase_shift = whether to use phase-shifted lattice
!  finite = whether to use finite (truncated) lattice
!
subroutine mstm_set_lattice(cell_wx, cell_wy, phase_shift, finite_lat)
    implicit none
    real(8), intent(in) :: cell_wx, cell_wy
    integer, intent(in) :: phase_shift, finite_lat
    periodic_lattice = .true.
    cell_width(1) = cell_wx
    cell_width(2) = cell_wy
    phase_shift_form = (phase_shift /= 0)
    finite_lattice = (finite_lat /= 0)
end subroutine mstm_set_lattice


!
!  Disable periodic lattice.
!
subroutine mstm_clear_lattice()
    implicit none
    periodic_lattice = .false.
    phase_shift_form = .false.
    finite_lattice = .false.
end subroutine mstm_clear_lattice


!
!  Prepare the calculation: compute host spheres, Mie coefficients,
!  translation orders, and allocate result arrays. Must be called after
!  mstm_set_spheres() and before mstm_solve().
!
subroutine mstm_prepare()
    implicit none
    integer :: i, n

    ! Determine host spheres (encapsulation)
    call findhostspheres()

    ! Initialize plane boundaries (must be called before sphere_layer_initialization)
    call plane_boundary_initialization()

    ! Assign spheres to layers
    call sphere_layer_initialization()

    ! Compute Mie coefficients and sphere orders
    call miecoefcalc(mie_epsilon)

    ! max_mie_order is now set by miecoefcalc
    ! Init special function tables
    call init(max_mie_order)

    ! Set cluster origin as centroid of external spheres
    cluster_origin(:) = 0.d0
    n = 0
    do i = 1, number_spheres
        if (host_sphere(i) == 0) then
            n = n + 1
            cluster_origin(:) = cluster_origin(:) + sphere_position(:,i)
        endif
    enddo
    if (n > 0) cluster_origin(:) = cluster_origin(:) / dble(n)

    ! Translation order computation
    if (allocated(translation_order)) deallocate(translation_order)
    allocate(translation_order(number_spheres))
    translation_order(1:number_spheres) = sphere_order(1:number_spheres)
    call tranorders(translation_epsilon, translation_order, t_matrix_order)
    t_matrix_order = min(t_matrix_order, max_t_matrix_order)

    ! Volumetric radius (only external spheres)
    vol_radius = 0.d0
    do i = 1, number_spheres
        if (host_sphere(i) == 0) then
            vol_radius = vol_radius + sphere_radius(i)**3
        endif
    enddo
    vol_radius = vol_radius**(1.d0/3.d0)

    ! Cross-section radius
    cross_section_radius = vol_radius

    ! Derive incident_sin_beta and incident_direction from incident_beta_deg,
    ! matching the authoritative CLI logic (mstm-input-37.f90 ~1310-1318):
    ! Snell's law at the incidence-side boundary layer, using whichever
    ! plane boundary (0 = medium, or the far boundary) the beam enters
    ! through, and deriving incident_direction from beta_deg rather than
    ! trusting a caller-supplied value. Deferred here (rather than done in
    ! mstm_set_incident) because layer_ref_index/number_plane_boundaries
    ! must already be finalized -- callers may configure layers before or
    ! after incidence. See mstm_set_incident's docstring for the latent bug
    ! this replaces.
    if (incident_beta_specified) then
        incident_beta = incident_beta_deg * pi / 180.d0
        if (incident_beta_deg <= 90.d0) then
            incident_direction = 1
            incident_sin_beta = dsin(incident_beta_deg * pi / 180.d0) &
                / dble(layer_ref_index(0))
        else
            incident_direction = 2
            incident_sin_beta = dsin(incident_beta_deg * pi / 180.d0) &
                / dble(layer_ref_index(number_plane_boundaries))
        endif
    else
        incident_beta = 0.d0
    endif

    ! Initialize incident field (sets incident_field_scale, etc.)
    call incident_field_initialization( &
        incident_alpha_deg * pi / 180.d0, &
        incident_sin_beta, &
        incident_direction)

    ! Set qeff_dim
    qeff_dim = 3

    ! Deallocate previous results if present
    if (allocated(q_eff)) deallocate(q_eff, q_vabs, q_eff_tot)
    allocate(q_eff(3, qeff_dim, number_spheres), &
        q_vabs(qeff_dim, number_spheres), &
        q_eff_tot(3, qeff_dim))
    q_eff = 0.d0
    q_vabs = 0.d0
    q_eff_tot = 0.d0

    ! Deallocate previous solution if present
    if (allocated(amnp_s)) deallocate(amnp_s)
    allocate(amnp_s(number_eqns, 2))
    amnp_s = (0.d0, 0.d0)

    if (allocated(amnp_0)) deallocate(amnp_0)
    allocate(amnp_0(2*t_matrix_order*(t_matrix_order+2), 2))
    amnp_0 = (0.d0, 0.d0)

    ! Allocate scattering matrix if needed
    if (calculate_scattering_matrix) then
        scat_mat_ldim = -scattering_map_dimension
        scat_mat_udim = scattering_map_dimension
        scat_mat_mdim = 16 + 16
        if (allocated(scat_mat)) deallocate(scat_mat)
        allocate(scat_mat(scat_mat_mdim, scat_mat_ldim:scat_mat_udim))
        scat_mat = 0.d0
    endif

end subroutine mstm_prepare


!
!  Solve for a fixed orientation.
!  Returns per-sphere efficiency factors and total efficiencies.
!
subroutine mstm_solve(q_ext, q_abs, q_sca, qext_tot, qabs_tot, &
        qsca_tot, sol_err, niter, status, n)
    implicit none
    ! n must be passed explicitly (the caller already knows it -- it's the
    ! same count passed to mstm_set_spheres). f2py's generated C glue
    ! cannot size an intent(out) array off a bare module variable reached
    ! transitively through `use` from a different compiled module
    ! (confirmed: both a direct `q_ext(number_spheres)` and an
    ! intent(hide)-with-default `n = number_spheres` produce the same C
    ! compile error, "number_spheres undeclared") -- only dimensioning off
    ! another dummy argument of the same subroutine works reliably.
    integer, intent(in) :: n
    real(8), intent(out) :: q_ext(n), q_abs(n), q_sca(n)
    real(8), intent(out) :: qext_tot, qabs_tot, qsca_tot
    real(8), intent(out) :: sol_err
    integer, intent(out) :: niter, status
    real(8) :: alpha, sinc
    integer :: dir, istat, i, sol_iter
    real(8) :: solerr
    integer, parameter :: qdim = 3

    alpha = incident_alpha_deg * pi / 180.d0
    sinc  = incident_sin_beta
    dir   = incident_direction

    if (.not. allocated(amnp_s)) allocate(amnp_s(number_eqns, 2))
    if (.not. allocated(q_eff)) allocate(q_eff(3, qdim, number_spheres))
    amnp_s = (0.d0, 0.d0)
    q_eff = 0.d0

    call fixedorsoln(alpha, sinc, dir, solution_epsilon, &
        max_iterations, amnp_s, q_eff, qdim, solerr, &
        sol_iter, 0, istat, &
        mpi_comm=mpi_comm_world, &
        excited_spheres=sphere_excitation_switch, &
        solution_method=solution_method(1:1), &
        initialize_solver=.true.)

    sol_err = solerr
    niter = sol_iter
    status = istat

    ! Per-sphere efficiencies
    do i = 1, n
        q_ext(i) = q_eff(1, 1, i)
        q_abs(i) = q_eff(2, 1, i)
        q_sca(i) = q_eff(3, 1, i)
    enddo

    ! Total efficiencies
    call qtotcalc(number_spheres, qdim, cross_section_radius, &
        q_eff, q_vabs, q_eff_tot)
    qext_tot = q_eff_tot(1, 1)
    qabs_tot = q_eff_tot(2, 1)
    qsca_tot = q_eff_tot(3, 1)

    ! Merge to common origin for scattering matrix
    if (single_origin_expansion .and. number_plane_boundaries == 0) then
        if (.not. allocated(amnp_0)) allocate(amnp_0(number_eqns, 2))
        amnp_0 = (0.d0, 0.d0)
        do i = 1, 2
            call merge_to_common_origin(t_matrix_order, amnp_s(:,i), &
                amnp_0(:,i), origin_position=cluster_origin, &
                merge_procs=.true., mpi_comm=mpi_comm_world)
        enddo
    endif

    ! If we have single origin + scattering matrix, compute it
    if (calculate_scattering_matrix .and. allocated(scat_mat)) then
        if (allocated(amnp_0)) then
            call scattering_matrix_calculation(amnp_0, scat_mat, &
                mpi_comm=mpi_comm_world)
        else
            call scattering_matrix_calculation(amnp_s, scat_mat, &
                mpi_comm=mpi_comm_world)
        endif
    endif

end subroutine mstm_solve


!
!  Get the full 4x4 Mueller matrix at a single (costheta, phi) angle.
!  Returns 16 real elements as a flat array (row-major relative to the
!  Fortran 4x4 layout -- unchanged from the ctypes-era convention).
!  mstm_solve() must have been called already.
!
!  NOTE: the full-grid scattering matrix (formerly mstm_get_smatrix_c,
!  which manually flattened scat_mat(scat_mat_mdim, scat_mat_ldim:udim)
!  into a 1D C array for ctypes to reshape on the Python side) has no
!  equivalent subroutine here. scat_mat and its bounds
!  (scat_mat_amin/amax/ldim/udim/mdim) are module-level variables declared
!  directly in inputinterface (no derived type involved), so f2py already
!  exposes them as plain Python attributes
!  (ext.inputinterface.scat_mat, .scat_mat_amin, ...) -- no wrapper needed.
!
subroutine mstm_scattering_angle(costheta, phi, sm)
    implicit none
    real(8), intent(in) :: costheta, phi
    real(8), intent(out) :: sm(16)
    complex(8) :: ampmat(2,2)
    logical :: singleorigin, iframe
    real(8) :: csca

    singleorigin = (number_plane_boundaries == 0 .and. single_origin_expansion)
    iframe = singleorigin .and. incident_frame

    if (allocated(amnp_0) .and. singleorigin) then
        if (azimuthal_average) then
            call numerical_sm_azimuthal_average_so(amnp_0, t_matrix_order, &
                costheta, sm, rotate_plane=iframe, normalize_s11=.false.)
        else
            call scatteringmatrix(amnp_0, t_matrix_order, costheta, phi, &
                ampmat, sm, rotate_plane=iframe, normalize_s11=.false.)
        endif
    else
        csca = pi * 2.d0
        call multiple_origin_scatteringmatrix(amnp_s, costheta, phi, &
            csca, ampmat, sm, rotate_plane=.true.)
    endif
end subroutine mstm_scattering_angle


!
!  Compute the T-matrix for the current cluster (writes/reads a temp file
!  internally, "tmatrix_temp.dat", identical to the pre-migration
!  behavior -- not something callers interact with directly).
!    n            -- number of spheres (for sizing q_ext/q_abs; see
!                     mstm_solve's docstring for why this can't be
!                     sized off the module's own number_spheres directly)
!    array_len    -- length of the flat tmatrix_data output array, i.e.
!                     2 * sum_{l=1}^{t_matrix_order} 2*(2l+1)*2*l*(l+2)
!                     (2 reals per complex entry). Callers should read
!                     t_matrix_order from ext.spheredata.t_matrix_order
!                     (set by mstm_prepare()) and compute this themselves
!                     -- same formula as the old mstm_get_tmatrix_size_c,
!                     which has no Fortran-side dependency beyond
!                     t_matrix_order and is better reimplemented in Python.
!
subroutine mstm_compute_tmatrix(tmatrix_data, tmat_order, q_ext, q_abs, &
        status, n, array_len)
    implicit none
    integer, intent(in) :: n, array_len
    real(8), intent(out) :: tmatrix_data(array_len)
    integer, intent(out) :: tmat_order, status
    real(8), intent(out) :: q_ext(n), q_abs(n)
    real(8), allocatable :: qeff_local(:,:)
    integer :: i, j, l, m, kq, p, ios
    real(8) :: re, im
    character*128 :: tmatfile

    tmatfile = 'tmatrix_temp.dat'
    t_matrix_output_file = tmatfile

    allocate(qeff_local(3, n))

    call tmatrix_solution( &
        solution_method=solution_method(1:1), &
        solution_eps=solution_epsilon, &
        convergence_eps=t_matrix_convergence_epsilon, &
        max_iterations=max_iterations, &
        t_matrix_file=tmatfile, &
        procs_per_soln=1, &
        mpi_comm=mpi_comm_world, &
        sphere_qeff=qeff_local, &
        solution_status=status, &
        sphere_excitation_list=sphere_excitation_switch)

    tmat_order = t_matrix_order

    do i = 1, n
        q_ext(i) = qeff_local(1, i)
        q_abs(i) = qeff_local(2, i)
    enddo

    deallocate(qeff_local)

    ! Read T-matrix from file into output array
    i = 1
    open(20, file=tmatfile, status='old', action='read', iostat=ios)
    if (ios /= 0) return

    ! Skip header (two integers: order, order)
    read(20, *, iostat=ios)
    if (ios /= 0) then
        close(20)
        return
    endif

    ! Read T-matrix entries in same order as written:
    ! for each source (l, k, q), then dest (n, m, p)
    ! format: ( 0.1234567890E+01, 0.1234567890E+01)
    do l = 1, tmat_order
        do kq = 1, 2*(2*l+1)
            do j = 1, l
                do m = -j, j
                    do p = 1, 2
                        read(20, '(1x,e18.10,1x,e18.10)', iostat=ios) re, im
                        if (ios /= 0) then
                            close(20)
                            return
                        endif
                        tmatrix_data(2*(i-1)+1) = re
                        tmatrix_data(2*(i-1)+2) = im
                        i = i + 1
                    enddo
                enddo
            enddo
        enddo
    enddo

    close(20)

end subroutine mstm_compute_tmatrix


!
!  Compute random orientation scattering matrix from a T-matrix file.
!  Output: sm_coef -- flat (16, 0:2*tmat_order) expansion coefficients for
!                      the 16 Mueller matrix elements
!          cm_coef -- same for coherent field
!          tmat_order_out -- order read from T-matrix file (should match
!                             the tmat_order passed in, sizing sm_coef/
!                             cm_coef; returned for confirmation)
!  tmatrix_file is a plain Fortran string here -- f2py marshals Python str
!  <-> Fortran character(len=*) natively, so the ctypes-era manual
!  byte-array + explicit length (c_len) conversion is no longer needed.
!
!  Two real bugs fixed here relative to the pre-migration
!  mstm_ranorient_smatrix_c (present in that version too, not introduced by
!  this port -- found while verifying this function against real data):
!  1. ranorientscatmatrix's tmatrixfile dummy argument is a FIXED
!     character*30 (mstm-scatprops-26.f90:1989), but the caller passed a
!     shorter string directly with no padding -- Fortran does not
!     auto-pad on argument association, so the callee read up to 30 bytes
!     starting at the caller's (shorter) buffer, into whatever memory
!     followed it. This is exactly the garbled-filename bug behind the
!     stray "tmatrix_temp.dat<garbage>" file seen in this project's
!     working tree. Fixed by explicitly assigning into a local
!     character(len=30) (Fortran space-pads on assignment to a longer
!     fixed-length variable) before the call.
!  2. override_order=0 was passed unconditionally. ranorientscatmatrix
!     only checks `present(override_order)` -- true regardless of the
!     value -- so this silently forced the T-matrix order to literally 0
!     instead of the order read from the file, making every returned
!     coefficient trivially zero. Fixed by omitting the optional argument
!     entirely so the file's own order is used.
!
subroutine mstm_ranorient_smatrix(tmatrix_file, sm_coef, cm_coef, &
        tmat_order_out, tmat_order)
    implicit none
    character(len=*), intent(in) :: tmatrix_file
    integer, intent(in) :: tmat_order
    real(8), intent(out) :: sm_coef(16*(2*tmat_order+1)), cm_coef(16*(2*tmat_order+1))
    integer, intent(out) :: tmat_order_out
    real(8), allocatable :: sm_local(:,:,:), cm_local(:,:,:)
    character(len=30) :: fname_padded
    integer :: i, j, n

    fname_padded = tmatrix_file

    allocate(sm_local(4, 4, 0:2*max_t_matrix_order), &
        cm_local(4, 4, 0:2*max_t_matrix_order))
    sm_local = 0.d0
    cm_local = 0.d0

    call ranorientscatmatrix(fname_padded, sm_local, cm_local, &
        beam_width=gaussian_beam_constant, &
        mpi_comm=mpi_comm_world, &
        keep_quiet=.true.)

    tmat_order_out = t_matrix_order

    ! Copy to flat output arrays
    do n = 0, 2*t_matrix_order
        do j = 1, 4
            do i = 1, 4
                sm_coef(16*n + 4*(j-1) + i) = sm_local(i, j, n)
                cm_coef(16*n + 4*(j-1) + i) = cm_local(i, j, n)
            enddo
        enddo
    enddo

    deallocate(sm_local, cm_local)

end subroutine mstm_ranorient_smatrix


!
!  Evaluate random orientation scattering matrix at a given costheta.
!  Uses pre-computed expansion coefficients (from mstm_ranorient_smatrix).
!
subroutine mstm_ranorient_smatrix_at_angle(sm_coef, tmat_order_in, &
        costheta, sm_elements)
    implicit none
    integer, intent(in) :: tmat_order_in
    real(8), intent(in) :: sm_coef(16*(2*tmat_order_in+1))
    real(8), intent(in) :: costheta
    real(8), intent(out) :: sm_elements(16)
    real(8), allocatable :: sm3d(:,:,:)
    integer :: n, i, j

    allocate(sm3d(4, 4, 0:2*tmat_order_in))
    sm3d = 0.d0
    do n = 0, 2*tmat_order_in
        do j = 1, 4
            do i = 1, 4
                sm3d(i, j, n) = sm_coef(16*n + 4*(j-1) + i)
            enddo
        enddo
    enddo

    ! This computes Mueller matrix from GSF expansion coefficients
    ! using the ranorienscatmatrixcalc subroutine
    call ranorienscatmatrixcalc(costheta, sm3d, tmat_order_in, sm_elements)

    deallocate(sm3d)

end subroutine mstm_ranorient_smatrix_at_angle


!
!  Set scattering map parameters for the full-range scattering matrix.
!    dim       -- half-range (total angles = 2*dim+1)
!    amin, amax -- angle range in degrees
!
subroutine mstm_set_scattering_map(dim, amin, amax)
    implicit none
    integer, intent(in) :: dim
    real(8), intent(in) :: amin, amax
    scattering_map_dimension = dim
    scat_mat_amin = amin
    scat_mat_amax = amax
    scat_mat_ldim = -dim
    scat_mat_udim = dim
    scat_mat_mdim = 32  ! 16 for up + 16 for down
    calculate_scattering_matrix = .true.
end subroutine mstm_set_scattering_map


!
!  Set per-sphere excitation switch.
!  excite(i) = 1 means sphere i is excited, 0 means not.
!  n must be passed explicitly -- see mstm_solve's docstring for why an
!  array can't be sized directly off number_spheres here.
!
subroutine mstm_set_excitation_switch(excite, n)
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: excite(n)
    integer :: i
    do i = 1, n
        sphere_excitation_switch(i) = (excite(i) /= 0)
    enddo
end subroutine mstm_set_excitation_switch


!
!  Set verbosity (0 = silent, 1 = verbose).
!
subroutine mstm_set_verbose(level)
    implicit none
    integer, intent(in) :: level
    if (level > 0) then
        print_intermediate_results = 1
        light_up = .true.
        print_timings = .true.
    else
        print_intermediate_results = 0
        light_up = .false.
        print_timings = .false.
    endif
end subroutine mstm_set_verbose


!
!  Finalize (MPI finalize, deallocate).
!
subroutine mstm_finalize()
    implicit none
    if (allocated(sphere_order)) &
        deallocate(sphere_order, sphere_radius, sphere_position, &
            sphere_ref_index, host_sphere, number_field_expansions, &
            sphere_excitation_switch, sphere_index, optically_active, &
            sphere_block, sphere_offset, mie_offset, qext_mie, qabs_mie)
    if (allocated(sphere_links)) deallocate(sphere_links)
    if (allocated(sphere_layer)) deallocate(sphere_layer)
    if (allocated(sphere_depth)) deallocate(sphere_depth)
    if (allocated(amnp_s)) deallocate(amnp_s)
    if (allocated(amnp_0)) deallocate(amnp_0)
    if (allocated(q_eff)) deallocate(q_eff, q_vabs, q_eff_tot)
    if (allocated(scat_mat)) deallocate(scat_mat)
    call mstm_mpi(mpi_command='finalize')
end subroutine mstm_finalize

end module mstm_f2py_bindings
