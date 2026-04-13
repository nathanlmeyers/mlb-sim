"""Baserunner state machine for MLB simulation.

Tracks runners on first, second, and third base.
Handles advancement on all plate appearance outcomes.
"""

from dataclasses import dataclass, field
import numpy as np
import config


@dataclass
class BaseState:
    """Tracks which bases are occupied.

    Each base holds a player_id (int) or None if empty.
    The 8 possible states (empty through bases loaded) combined with
    0/1/2 outs give 24 distinct game micro-states per half-inning.
    """
    first: int | None = None
    second: int | None = None
    third: int | None = None

    def is_empty(self) -> bool:
        return self.first is None and self.second is None and self.third is None

    def runners_on(self) -> int:
        """Count of runners on base."""
        return sum(1 for b in (self.first, self.second, self.third) if b is not None)

    def encode(self) -> str:
        """Encode base state as string like '1--', '-2-', '123', '---'."""
        f = "1" if self.first is not None else "-"
        s = "2" if self.second is not None else "-"
        t = "3" if self.third is not None else "-"
        return f"{f}{s}{t}"

    def advance_on_walk(self, batter_id: int) -> int:
        """Advance forced runners on walk/HBP. Returns runs scored."""
        runs = 0
        # Only forced runners advance
        if self.first is not None:
            if self.second is not None:
                if self.third is not None:
                    runs += 1  # runner on 3rd forced home
                self.third = self.second
            self.second = self.first
        self.first = batter_id
        return runs

    def advance_on_single(self, batter_id: int, batter_speed: float, rng: np.random.Generator) -> int:
        """Advance runners on single. Probabilistic based on speed."""
        runs = 0

        # Runner on 3rd: scores ~95%
        if self.third is not None:
            if rng.random() < config.P_SCORE_FROM_THIRD_ON_SINGLE:
                runs += 1
            # If doesn't score, stays (very rare, treat as scoring)
            else:
                runs += 1
            self.third = None

        # Runner on 2nd: scores ~45%
        if self.second is not None:
            if rng.random() < config.P_SCORE_FROM_SECOND_ON_SINGLE:
                runs += 1
                self.second = None
            else:
                self.third = self.second
                self.second = None

        # Runner on 1st: to 3rd ~28%, to 2nd ~72%
        if self.first is not None:
            if rng.random() < config.P_ADVANCE_FIRST_TO_THIRD_ON_SINGLE:
                if self.third is None:
                    self.third = self.first
                else:
                    # 3rd occupied, runner stays at 2nd
                    self.second = self.first
            else:
                self.second = self.first
            self.first = None

        # Batter to first
        self.first = batter_id
        return runs

    def advance_on_double(self, batter_id: int, batter_speed: float, rng: np.random.Generator) -> int:
        """Advance runners on double."""
        runs = 0

        # Runner on 3rd: scores
        if self.third is not None:
            runs += 1
            self.third = None

        # Runner on 2nd: scores ~95%
        if self.second is not None:
            if rng.random() < config.P_SCORE_FROM_SECOND_ON_DOUBLE:
                runs += 1
            else:
                self.third = self.second
            self.second = None

        # Runner on 1st: scores ~44%, to 3rd ~56%
        if self.first is not None:
            if rng.random() < config.P_SCORE_FROM_FIRST_ON_DOUBLE:
                runs += 1
            else:
                self.third = self.first
            self.first = None

        # Batter to second
        self.second = batter_id
        return runs

    def advance_on_triple(self, batter_id: int) -> int:
        """Advance runners on triple. All runners score."""
        runs = 0
        if self.third is not None:
            runs += 1
            self.third = None
        if self.second is not None:
            runs += 1
            self.second = None
        if self.first is not None:
            runs += 1
            self.first = None
        # Batter to third
        self.third = batter_id
        return runs

    def advance_on_hr(self) -> int:
        """Advance runners on home run. All runners + batter score."""
        runs = 1  # batter scores
        if self.third is not None:
            runs += 1
        if self.second is not None:
            runs += 1
        if self.first is not None:
            runs += 1
        self.first = None
        self.second = None
        self.third = None
        return runs

    def advance_on_groundout(self, outs: int, rng: np.random.Generator) -> tuple[int, int]:
        """Handle ground ball out. Returns (runs_scored, outs_recorded).

        Checks for double play (runner on 1st, <2 outs).
        """
        runs = 0
        outs_recorded = 1

        # Double play check: runner on 1st and <2 outs
        if self.first is not None and outs < 2:
            if rng.random() < config.P_DOUBLE_PLAY_ON_GB:
                outs_recorded = 2
                # Runner on 1st is out, batter is out
                self.first = None
                # Runner on 3rd may score on DP
                if self.third is not None:
                    runs += 1
                    self.third = None
                # Runner on 2nd advances to 3rd
                if self.second is not None:
                    self.third = self.second
                    self.second = None
                return runs, outs_recorded

        # Regular groundout: runner on 3rd may score if <2 outs
        if self.third is not None and outs < 2:
            if rng.random() < 0.50:  # groundout to infield, runner holds/scores
                runs += 1
                self.third = None

        # Runners advance one base on groundout
        if self.second is not None and self.third is None:
            self.third = self.second
            self.second = None
        if self.first is not None and self.second is None:
            self.second = self.first
            self.first = None

        return runs, outs_recorded

    def advance_on_flyout(self, outs: int, rng: np.random.Generator) -> tuple[int, int]:
        """Handle fly ball out. Returns (runs_scored, outs_recorded).

        Checks for sacrifice fly (runner on 3rd, <2 outs).
        """
        runs = 0
        outs_recorded = 1

        # Sac fly: runner on 3rd scores if <2 outs
        if self.third is not None and outs < 2:
            if rng.random() < config.P_SAC_FLY:
                runs += 1
                self.third = None

        return runs, outs_recorded

    def place_manfred_runner(self, runner_id: int) -> None:
        """Place the Manfred runner on 2nd base for extra innings."""
        self.second = runner_id
