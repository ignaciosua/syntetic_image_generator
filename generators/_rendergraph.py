"""Dependency graph for heterogeneous bulk rendering.

The graph is deliberately small and backend-agnostic.  It does not try to be
an HVM-style evaluator; instead it captures the useful part of that model for
this library: independent work is explicit, dependencies are validated, and
the scheduler can submit CPU preparation before unrelated bulk CPU rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RenderNode:
    """One homogeneous stage in a compiled bulk render graph."""

    key: str
    family: str
    stage: str
    positions: tuple[int, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderGraph:
    """An immutable, topologically validated bulk execution plan."""

    nodes: tuple[RenderNode, ...]

    @classmethod
    def compile(
        cls,
        levels: Sequence[int],
        family_by_level: Mapping[int, str],
        stages_by_family: Mapping[str, Sequence[str]],
        *,
        enabled_families: Iterable[str] | None = None,
    ) -> "RenderGraph":
        """Group batch positions and compile their stage dependencies."""
        enabled = (
            set(stages_by_family)
            if enabled_families is None
            else set(enabled_families)
        )
        unknown = enabled.difference(stages_by_family)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown render graph families: {names}")

        family_positions: dict[str, list[int]] = {}
        cpu_positions: list[int] = []
        for position, level in enumerate(levels):
            family = family_by_level.get(int(level))
            if family is None or family not in enabled:
                cpu_positions.append(position)
            else:
                family_positions.setdefault(family, []).append(position)

        nodes: list[RenderNode] = []
        terminal_keys: list[str] = []
        # Stage-map insertion order is the scheduling policy.  It lets callers
        # put a preparation-free family first so the GPU starts immediately
        # while CPU workers prepare later graph branches.
        for family, stages in stages_by_family.items():
            positions = family_positions.get(family)
            if not positions:
                continue
            previous: tuple[str, ...] = ()
            for stage in stages:
                key = f"{family}:{stage}"
                nodes.append(
                    RenderNode(
                        key=key,
                        family=family,
                        stage=str(stage),
                        positions=tuple(positions),
                        dependencies=previous,
                    )
                )
                previous = (key,)
            if not previous:
                raise ValueError(
                    f"render graph family {family!r} has no stages"
                )
            terminal_keys.extend(previous)

        if cpu_positions:
            cpu_key = "cpu:render"
            nodes.append(
                RenderNode(
                    key=cpu_key,
                    family="cpu",
                    stage="render",
                    positions=tuple(cpu_positions),
                )
            )
            terminal_keys.append(cpu_key)

        nodes.append(
            RenderNode(
                key="batch:assemble",
                family="batch",
                stage="assemble",
                positions=tuple(range(len(levels))),
                dependencies=tuple(terminal_keys),
            )
        )
        graph = cls(tuple(nodes))
        graph.topological_nodes()
        return graph

    def node(self, key: str) -> RenderNode:
        for node in self.nodes:
            if node.key == key:
                return node
        raise KeyError(key)

    def positions(self, family: str) -> tuple[int, ...]:
        """Return positions owned by a family, independent of its stage."""
        for node in self.nodes:
            if node.family == family:
                return node.positions
        return ()

    @property
    def gpu_families(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                node.family
                for node in self.nodes
                if node.family not in {"cpu", "batch"}
            )
        )

    def topological_nodes(self) -> tuple[RenderNode, ...]:
        """Return a stable topological order, rejecting malformed graphs."""
        by_key = {node.key: node for node in self.nodes}
        if len(by_key) != len(self.nodes):
            raise ValueError("render graph node keys must be unique")
        for node in self.nodes:
            missing = set(node.dependencies).difference(by_key)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"render graph node {node.key!r} has missing "
                    f"dependencies: {names}"
                )

        remaining = {node.key: set(node.dependencies) for node in self.nodes}
        ordered: list[RenderNode] = []
        while remaining:
            ready = [
                node
                for node in self.nodes
                if node.key in remaining and not remaining[node.key]
            ]
            if not ready:
                raise ValueError("render graph contains a dependency cycle")
            for node in ready:
                ordered.append(node)
                del remaining[node.key]
                for dependencies in remaining.values():
                    dependencies.discard(node.key)
        return tuple(ordered)

