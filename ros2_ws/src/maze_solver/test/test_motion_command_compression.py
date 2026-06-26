import math

import pytest

from maze_solver_node import build_motion_commands


def assert_commands_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for (actual_name, actual_value), (expected_name, expected_value) in zip(
        actual,
        expected,
    ):
        assert actual_name == expected_name
        assert actual_value == pytest.approx(expected_value, abs=tol)


def test_straight_three_cells_split_into_two_plus_one():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 1, 1],
        cell_length_m=0.254,
        max_cells_per_drive=2,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.508),
        ('drive_forward', 0.254),
    ])


def test_right_angle_then_two_cells():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 4, 4],
        cell_length_m=0.254,
        max_cells_per_drive=2,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.254),
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.508),
    ])


def test_left_turn_back_to_forward():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[4, 4, 1],
        cell_length_m=0.254,
        max_cells_per_drive=2,
    )

    assert_commands_close(commands, [
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.508),
        ('rotate', -math.pi / 2.0),
        ('drive_forward', 0.254),
    ])


def test_max_cells_per_drive_one_preserves_one_cell_commands():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 1],
        cell_length_m=0.254,
        max_cells_per_drive=1,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.254),
        ('drive_forward', 0.254),
    ])
