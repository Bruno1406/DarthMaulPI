from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


MAX_CUBES = 4

ADD_RESULT_ADDED = 'added'
ADD_RESULT_UPDATED = 'updated'
ADD_RESULT_CAPACITY = 'capacity'


@dataclass
class Cube:
    x_cm: float
    y_cm: float
    color_votes: Dict[int, int] = field(default_factory=dict)
    seen_count: int = 0

    def __init__(
        self,
        x_cm: float,
        y_cm: float,
        color_id: int,
    ) -> None:
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)
        self.color_votes = {}
        self.seen_count = 1
        self.add_color_vote(color_id)

    def add_color_vote(self, color_id: int) -> None:
        normalized_color = int(color_id)
        self.color_votes[normalized_color] = (
            self.color_votes.get(normalized_color, 0) + 1
        )

    def distance_to(
        self,
        x_cm: float,
        y_cm: float,
    ) -> float:
        dx = self.x_cm - float(x_cm)
        dy = self.y_cm - float(y_cm)
        return (dx * dx + dy * dy) ** 0.5

    def update(
        self,
        x_cm: float,
        y_cm: float,
        color_id: int,
    ) -> None:
        old_count = self.seen_count
        self.seen_count += 1

        self.x_cm = (
            self.x_cm * old_count + float(x_cm)
        ) / self.seen_count

        self.y_cm = (
            self.y_cm * old_count + float(y_cm)
        ) / self.seen_count

        self.add_color_vote(color_id)

    def get_color_id(self) -> int:
        return int(max(self.color_votes, key=self.color_votes.get))


class CubeTracker:
    def __init__(
        self,
        limit_distance_cm: float = 20.0,
        max_cubes: int = MAX_CUBES,
    ) -> None:
        self.cubes: List[Cube] = []
        self.limit_distance_cm = float(limit_distance_cm)

        # This is deliberately capped at four, regardless of caller input.
        self.max_cubes = min(
            MAX_CUBES,
            max(1, int(max_cubes)),
        )

    def add_cube(
        self,
        x_cm: float,
        y_cm: float,
        color_id: int,
    ) -> str:
        nearest_cube = None
        nearest_distance = float('inf')

        for cube in self.cubes:
            distance = cube.distance_to(x_cm, y_cm)

            if distance < nearest_distance:
                nearest_cube = cube
                nearest_distance = distance

        if (
            nearest_cube is not None
            and nearest_distance <= self.limit_distance_cm
        ):
            nearest_cube.update(
                x_cm,
                y_cm,
                color_id,
            )
            return ADD_RESULT_UPDATED

        if len(self.cubes) >= self.max_cubes:
            return ADD_RESULT_CAPACITY

        self.cubes.append(
            Cube(
                x_cm,
                y_cm,
                color_id,
            )
        )
        return ADD_RESULT_ADDED

    def snapshot(
        self,
    ) -> List[Tuple[float, float, int, int]]:
        return [
            (
                float(cube.x_cm),
                float(cube.y_cm),
                int(cube.get_color_id()),
                int(cube.seen_count),
            )
            for cube in self.cubes[:MAX_CUBES]
        ]

    def export_for_service(
        self,
    ) -> Tuple[int, List[int], List[int], List[int]]:
        cubes = self.snapshot()

        xs = [
            int(round(cube[0]))
            for cube in cubes
        ]
        ys = [
            int(round(cube[1]))
            for cube in cubes
        ]
        colors = [
            int(cube[2])
            for cube in cubes
        ]

        return len(cubes), xs, ys, colors

    def get_cube_count(self) -> int:
        return min(
            len(self.cubes),
            MAX_CUBES,
        )
