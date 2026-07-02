import math
from pathlib import Path

import pytest

from maze_solver_node import (
    build_motion_commands,
    parse_bool_parameter,
    validate_maze_fields,
    validate_solver_parameters,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REMOVED_CELL_CAP_PARAM = 'max_cells' + '_per_drive'


def assert_commands_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for (actual_name, actual_value), (expected_name, expected_value) in zip(
        actual,
        expected,
    ):
        assert actual_name == expected_name
        assert actual_value == pytest.approx(expected_value, abs=tol)


def test_straight_three_cells_one_continuous_drive():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 1, 1],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.762),
    ])


def test_five_cell_straight_is_one_continuous_drive():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 1, 1, 1, 1],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('drive_forward', 1.270),
    ])


def test_empty_orientation_list_produces_no_motion_commands():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [])


def test_turn_then_continuous_straight_run():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 4, 4],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.254),
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.508),
    ])


def test_multiple_straight_runs_are_split_only_by_turns():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[1, 1, 4, 4, 4, 2, 2],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.508),
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.762),
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.508),
    ])


def test_negative_turn_between_runs_is_preserved():
    commands = build_motion_commands(
        start_orientation=4,
        orientations=[4, 1, 1],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('drive_forward', 0.254),
        ('rotate', -math.pi / 2.0),
        ('drive_forward', 0.508),
    ])


def test_left_turn_back_to_forward():
    commands = build_motion_commands(
        start_orientation=1,
        orientations=[4, 4, 1],
        cell_length_m=0.254,
    )

    assert_commands_close(commands, [
        ('rotate', math.pi / 2.0),
        ('drive_forward', 0.508),
        ('rotate', -math.pi / 2.0),
        ('drive_forward', 0.254),
    ])


def test_parse_bool_parameter_accepts_common_false_strings():
    assert parse_bool_parameter('false') is False
    assert parse_bool_parameter('False') is False
    assert parse_bool_parameter('0') is False
    assert parse_bool_parameter('no') is False
    assert parse_bool_parameter('off') is False


def test_parse_bool_parameter_accepts_common_true_strings():
    assert parse_bool_parameter('true') is True
    assert parse_bool_parameter('True') is True
    assert parse_bool_parameter('1') is True
    assert parse_bool_parameter('yes') is True
    assert parse_bool_parameter('on') is True


def test_validate_maze_fields_accepts_valid_minimal_maze():
    assert validate_maze_fields(
        n=2,
        m=2,
        start_idx=1,
        end_idx=4,
        start_orientation=1,
        flattened_l=[10, 9, 6, 5],
    ) is None


def test_validate_maze_fields_rejects_wrong_flattened_length():
    error = validate_maze_fields(
        n=2,
        m=2,
        start_idx=1,
        end_idx=4,
        start_orientation=1,
        flattened_l=[10, 9, 6],
    )

    assert error is not None
    assert 'invalid maze length' in error


def test_validate_maze_fields_rejects_invalid_start_orientation():
    error = validate_maze_fields(
        n=2,
        m=2,
        start_idx=1,
        end_idx=4,
        start_orientation=3,
        flattened_l=[10, 9, 6, 5],
    )

    assert error is not None
    assert 'invalid start_orientation' in error


def test_validate_maze_fields_rejects_missing_start_or_end_cell():
    start_error = validate_maze_fields(
        n=2,
        m=2,
        start_idx=1,
        end_idx=4,
        start_orientation=1,
        flattened_l=[15, 9, 6, 5],
    )
    end_error = validate_maze_fields(
        n=2,
        m=2,
        start_idx=1,
        end_idx=4,
        start_orientation=1,
        flattened_l=[10, 9, 6, 15],
    )

    assert start_error is not None
    assert 'start_idx' in start_error
    assert end_error is not None
    assert 'end_idx' in end_error


def test_validate_solver_parameters_accepts_exam_defaults():
    assert validate_solver_parameters(
        maze_nr=1,
        cell_length_m=0.254,
        max_commands_to_execute=0,
        motion_server_timeout_s=5.0,
        maze_service_timeout_s=15.0,
        maze_service_name='/get_ros_maze',
    ) is None


def test_validate_solver_parameters_rejects_bad_maze_nr():
    error = validate_solver_parameters(
        maze_nr=200,
        cell_length_m=0.254,
        max_commands_to_execute=0,
        motion_server_timeout_s=5.0,
        maze_service_timeout_s=15.0,
        maze_service_name='/get_ros_maze',
    )

    assert error is not None
    assert 'int8 range' in error


def test_validate_solver_parameters_rejects_bad_cell_length():
    for bad_value in (0.0, -0.1, float('nan'), float('inf')):
        error = validate_solver_parameters(
            maze_nr=1,
            cell_length_m=bad_value,
            max_commands_to_execute=0,
            motion_server_timeout_s=5.0,
            maze_service_timeout_s=15.0,
            maze_service_name='/get_ros_maze',
        )

        assert error is not None
        assert 'cell_length_m' in error


def test_validate_solver_parameters_rejects_bad_max_commands_to_execute():
    error = validate_solver_parameters(
        maze_nr=1,
        cell_length_m=0.254,
        max_commands_to_execute=-1,
        motion_server_timeout_s=5.0,
        maze_service_timeout_s=15.0,
        maze_service_name='/get_ros_maze',
    )

    assert error is not None
    assert 'max_commands_to_execute' in error


def test_validate_solver_parameters_rejects_bad_motion_timeout():
    for bad_value in (0.0, -1.0, float('nan'), float('inf')):
        error = validate_solver_parameters(
            maze_nr=1,
            cell_length_m=0.254,
            max_commands_to_execute=0,
            motion_server_timeout_s=bad_value,
            maze_service_timeout_s=15.0,
            maze_service_name='/get_ros_maze',
        )

        assert error is not None
        assert 'motion_server_timeout_s' in error


def test_validate_solver_parameters_rejects_bad_maze_timeout():
    for bad_value in (0.0, -1.0, float('nan'), float('inf')):
        error = validate_solver_parameters(
            maze_nr=1,
            cell_length_m=0.254,
            max_commands_to_execute=0,
            motion_server_timeout_s=5.0,
            maze_service_timeout_s=bad_value,
            maze_service_name='/get_ros_maze',
        )

        assert error is not None
        assert 'maze_service_timeout_s' in error


def test_validate_solver_parameters_rejects_relative_maze_service_name():
    error = validate_solver_parameters(
        maze_nr=1,
        cell_length_m=0.254,
        max_commands_to_execute=0,
        motion_server_timeout_s=5.0,
        maze_service_timeout_s=15.0,
        maze_service_name='get_ros_maze',
    )

    assert error is not None
    assert 'absolute service name' in error


def test_exam_launch_has_no_removed_cell_cap_argument():
    launch_text = (
        PACKAGE_ROOT / 'launch' / 'exam_task1.launch.py'
    ).read_text()

    assert REMOVED_CELL_CAP_PARAM not in launch_text
    assert 'max_commands_to_execute' in launch_text
    assert 'execute_motions' in launch_text
    assert 'cell_length_m' in launch_text


def test_solver_node_has_no_removed_cell_cap_parameter():
    solver_text = (PACKAGE_ROOT / 'maze_solver_node.py').read_text()

    assert REMOVED_CELL_CAP_PARAM not in solver_text
