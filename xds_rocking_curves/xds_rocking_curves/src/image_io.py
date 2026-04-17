"""Image IO and patch extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import re

import numpy as np
import tifffile


@dataclass(frozen=True)
class PatchExtraction:
    """Patch extraction result."""

    patch: np.ndarray
    x_start_zero: int
    y_start_zero: int
    full_inside: bool


class ImageResolver:
    """Resolve frame numbers to image paths.

    Supports either a sorted image glob or an XDS ``NAME_TEMPLATE_OF_DATA_FRAMES``
    style template with ``?`` placeholders.
    """

    def __init__(
        self,
        image_glob: str | None = None,
        image_template: str | None = None,
        starting_frame: int = 1,
        frame_range: tuple[int, int] | None = None,
    ) -> None:
        if image_glob is None and image_template is None:
            raise ValueError("Either image_glob or image_template must be supplied.")
        self._starting_frame = int(starting_frame)
        self._frame_to_path: dict[int, Path] = {}

        if image_glob is not None:
            paths = sorted(Path(p) for p in glob.glob(image_glob))
            if not paths:
                raise FileNotFoundError(f"Image glob matched no files: {image_glob}")
            for idx, path in enumerate(paths, start=self._starting_frame):
                self._frame_to_path[idx] = path
        else:
            template_path = Path(image_template).expanduser()
            template_str = str(template_path)
            if "?" not in template_str:
                raise ValueError("image_template must contain ? placeholders when no image_glob is used.")
            q_count = template_str.count("?")
            regex = re.compile(re.escape(template_str).replace("\\?" * q_count, rf"(?P<num>\d{{{q_count}}})"))
            candidate_glob = template_str.replace("?" * q_count, "*" * q_count)
            paths = sorted(Path(p) for p in glob.glob(candidate_glob))
            if not paths:
                raise FileNotFoundError(f"Template matched no files: {image_template}")
            for path in paths:
                match = regex.fullmatch(str(path))
                if match is None:
                    continue
                frame = int(match.group("num"))
                self._frame_to_path[frame] = path
            if not self._frame_to_path:
                raise FileNotFoundError(f"Could not build a frame map from template: {image_template}")

        if frame_range is not None:
            frame_lo, frame_hi = int(frame_range[0]), int(frame_range[1])
            if frame_lo > frame_hi:
                raise ValueError("frame_range must satisfy frame_min <= frame_max")
            self._frame_to_path = {
                frame: path for frame, path in self._frame_to_path.items() if frame_lo <= frame <= frame_hi
            }
            if not self._frame_to_path:
                raise FileNotFoundError(
                    f"No images remained after applying frame_range=({frame_lo}, {frame_hi})."
                )

    @property
    def available_frames(self) -> list[int]:
        """Return available frame numbers."""

        return sorted(self._frame_to_path)

    def path_for_frame(self, frame: int) -> Path:
        """Return the file path for a frame."""

        frame_i = int(frame)
        if frame_i not in self._frame_to_path:
            raise KeyError(f"No image path known for frame {frame_i}.")
        return self._frame_to_path[frame_i]


def read_image(path: str | Path) -> np.ndarray:
    """Read an image using tifffile and return float64 pixels."""

    array = tifffile.imread(path)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D diffraction image, got shape {array.shape} from {path}")
    return np.asarray(array, dtype=np.float64)


def extract_patch(
    image: np.ndarray,
    x_pred_px: float,
    y_pred_px: float,
    half_size: int,
) -> PatchExtraction:
    """Extract a square patch around an XDS pixel coordinate.

    XDS coordinates are treated as 1-based pixel-center coordinates, so they are
    shifted to zero-based numpy coordinates before slicing.
    """

    if half_size < 1:
        raise ValueError("half_size must be at least 1 pixel.")

    x_zero = float(x_pred_px) - 1.0
    y_zero = float(y_pred_px) - 1.0
    x_start = int(np.floor(x_zero)) - int(half_size)
    y_start = int(np.floor(y_zero)) - int(half_size)
    size = 2 * int(half_size) + 1
    x_stop = x_start + size
    y_stop = y_start + size

    full_inside = x_start >= 0 and y_start >= 0 and x_stop <= image.shape[1] and y_stop <= image.shape[0]
    if full_inside:
        patch = image[y_start:y_stop, x_start:x_stop].copy()
        return PatchExtraction(patch=patch, x_start_zero=x_start, y_start_zero=y_start, full_inside=True)

    patch = np.full((size, size), np.nan, dtype=np.float64)
    src_x0 = max(0, x_start)
    src_y0 = max(0, y_start)
    src_x1 = min(image.shape[1], x_stop)
    src_y1 = min(image.shape[0], y_stop)
    dst_x0 = src_x0 - x_start
    dst_y0 = src_y0 - y_start
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    patch[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return PatchExtraction(patch=patch, x_start_zero=x_start, y_start_zero=y_start, full_inside=False)


def predicted_center_in_patch(
    x_pred_px: float,
    y_pred_px: float,
    extraction: PatchExtraction,
) -> tuple[float, float]:
    """Return the predicted center within a patch in zero-based patch coordinates."""

    x_local = (float(x_pred_px) - 1.0) - float(extraction.x_start_zero)
    y_local = (float(y_pred_px) - 1.0) - float(extraction.y_start_zero)
    return x_local, y_local
