"""PyTorch dataset wrappers for L2T examples."""

from __future__ import annotations

from collections.abc import Sequence

from .gridworld import GridExample
from .arithmetic import ArithmeticExample


class GridWorldTorchDataset:
    def __init__(self, examples: Sequence[GridExample], *, max_program_len: int = 8) -> None:
        import torch

        self.examples = list(examples)
        self._torch = torch
        self.max_program_len = max_program_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = self._torch
        item = self.examples[index]
        padded_program = list(item.program[: self.max_program_len])
        program_len = len(padded_program)
        padded_program.extend([-1] * (self.max_program_len - program_len))
        return {
            "support_x": torch.tensor(item.support_x, dtype=torch.long),
            "support_y": torch.tensor(item.support_y, dtype=torch.long),
            "query_x": torch.tensor(item.query_x, dtype=torch.long),
            "query_y": torch.tensor(item.query_y, dtype=torch.long),
            "program": torch.tensor(padded_program, dtype=torch.long),
            "program_len": torch.tensor(program_len, dtype=torch.long),
        }


class ArithmeticTorchDataset:
    def __init__(self, examples: Sequence[ArithmeticExample], *, max_program_len: int = 8) -> None:
        import torch

        self.examples = list(examples)
        self._torch = torch
        self.max_program_len = max_program_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch = self._torch
        item = self.examples[index]
        padded_program = list(item.program[: self.max_program_len])
        program_len = len(padded_program)
        padded_program.extend([-1] * (self.max_program_len - program_len))
        return {
            "support_x": torch.tensor(item.support_x_exp, dtype=torch.long),
            "support_y": torch.tensor(item.support_y_exp, dtype=torch.long),
            "query_x": torch.tensor(item.query_x_exp, dtype=torch.long),
            "query_y": torch.tensor(item.query_y_exp, dtype=torch.long),
            "support_x_value": torch.tensor(item.support_x, dtype=torch.long),
            "support_y_value": torch.tensor(item.support_y, dtype=torch.long),
            "query_x_value": torch.tensor(item.query_x, dtype=torch.long),
            "query_y_value": torch.tensor(item.query_y, dtype=torch.long),
            "program": torch.tensor(padded_program, dtype=torch.long),
            "program_len": torch.tensor(program_len, dtype=torch.long),
        }
