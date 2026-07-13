from darth_maul_control.scan_geometry import choose_lidar_progress


def test_rejects_impossible_rear_surface_transition_before_arbitration():
    estimate = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.276,
        rear_valid=True,
        rear_progress_m=1.517,
        max_disagreement_m=0.100,
        min_progress_m=-0.020,
        allow_single_source=True,
        odom_progress_m=0.128,
        odom_arbitration_tolerance_m=0.100,
        odom_arbitration_min_margin_m=0.010,
        max_progress_m=0.381,
    )

    assert estimate.valid
    assert estimate.source == 'front'
    assert estimate.progress_m == 0.276
    assert 'likely range-surface transition' in estimate.reason
    assert 'rear progress 1.517 m > maximum plausible 0.381 m' in estimate.reason


def test_maximum_plausible_progress_is_inclusive():
    estimate = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.381,
        rear_valid=False,
        rear_progress_m=0.0,
        max_disagreement_m=0.100,
        min_progress_m=-0.020,
        allow_single_source=True,
        max_progress_m=0.381,
    )

    assert estimate.valid
    assert estimate.source == 'front'
    assert estimate.progress_m == 0.381


def test_rejects_all_sources_above_maximum_plausible_progress():
    estimate = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.500,
        rear_valid=True,
        rear_progress_m=0.600,
        max_disagreement_m=0.100,
        min_progress_m=-0.020,
        allow_single_source=True,
        max_progress_m=0.381,
    )

    assert not estimate.valid
    assert estimate.source == 'none'
    assert estimate.reason.count('likely range-surface transition') == 2
