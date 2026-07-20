from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "hair": (38, 70, 83),
    "brows": (42, 157, 143),
    "eyes": (230, 130, 35),
    "glasses": (233, 196, 106),
    "nose": (244, 162, 97),
    "mouth": (231, 111, 81),
    "ears": (90, 120, 210),
    "face_outline": (135, 95, 170),
    "neck_clothes": (80, 150, 95),
}


def rect_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[max(0, y0):min(shape[0], y1), max(0, x0):min(shape[1], x1)] = 255
    return mask


def poly_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def build_regions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    sx = w / 512.0
    sy = h / 512.0

    def srect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return rect_mask(shape, int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

    def spoly(points: list[tuple[int, int]]) -> np.ndarray:
        return poly_mask(shape, [(int(x * sx), int(y * sy)) for x, y in points])

    left_lens = spoly([(122, 182), (248, 184), (238, 252), (158, 255), (132, 238)])
    right_lens = spoly([(270, 184), (396, 182), (386, 238), (358, 255), (280, 252)])
    bridge = srect(234, 188, 286, 215)
    glasses = left_lens | right_lens | bridge

    return {
        "hair": spoly([(80, 0), (430, 0), (430, 185), (350, 155), (260, 118), (160, 175), (80, 210)]),
        "brows": srect(145, 145, 365, 184),
        "eyes": srect(155, 188, 363, 224),
        "glasses": glasses,
        "nose": spoly([(220, 230), (295, 230), (315, 292), (255, 305), (202, 292)]),
        "mouth": srect(195, 305, 320, 370),
        "ears": spoly([(92, 188), (142, 188), (153, 303), (112, 318), (92, 258)])
        | spoly([(370, 188), (420, 188), (420, 258), (400, 318), (360, 303)]),
        "face_outline": spoly([(85, 70), (425, 70), (420, 375), (345, 440), (255, 460), (165, 430), (92, 330)])
        & cv2.bitwise_not(glasses),
        "neck_clothes": spoly([(0, 365), (512, 365), (512, 512), (0, 512)]),
    }


def connected_component_filter(
    ink: np.ndarray,
    region: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
) -> np.ndarray:
    candidate = cv2.bitwise_and(ink, region)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    out = np.zeros_like(ink)
    for label in range(1, num_labels):
        _, _, width, height, area = stats[label]
        if (
            min_area <= area <= max_area
            and min_width <= width <= max_width
            and min_height <= height <= max_height
        ):
            out[labels == label] = 255
    return out


def remove_assigned(mask: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(mask, cv2.bitwise_not(assigned))


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise SystemExit(f"Could not read image: {input_path}")

    ink = (gray < 210).astype(np.uint8) * 255
    regions = build_regions(gray.shape)

    eyes_mask = connected_component_filter(
        ink,
        regions["eyes"],
        min_area=20,
        max_area=900,
        min_width=4,
        max_width=80,
        min_height=3,
        max_height=32,
    )

    explicit_masks = {
        "eyes": eyes_mask,
        "glasses": cv2.bitwise_and(ink, cv2.bitwise_and(regions["glasses"], cv2.bitwise_not(eyes_mask))),
    }

    # Order matters: detailed facial features should claim ink before broad regions.
    priority = [
        "eyes",
        "glasses",
        "brows",
        "nose",
        "mouth",
        "ears",
        "hair",
        "neck_clothes",
        "face_outline",
    ]

    assigned = np.zeros_like(ink)
    masks: dict[str, np.ndarray] = {}
    for name in priority:
        candidate = explicit_masks.get(name, cv2.bitwise_and(ink, regions[name]))
        semantic = remove_assigned(candidate, assigned)
        masks[name] = semantic
        assigned = cv2.bitwise_or(assigned, semantic)

    remainder = remove_assigned(ink, assigned)
    masks["remainder"] = remainder

    overlay = np.full_like(color, 248)
    overlay[ink > 0] = (190, 190, 190)
    alpha = 0.78
    for name in priority + ["remainder"]:
        mask = masks[name]
        if not np.any(mask):
            continue
        rgb = COLORS.get(name, (100, 100, 100))
        bgr = np.array((rgb[2], rgb[1], rgb[0]), dtype=np.uint8)
        overlay[mask > 0] = (overlay[mask > 0] * (1.0 - alpha) + bgr * alpha).astype(np.uint8)

    legend_x, legend_y = 12, 18
    for index, name in enumerate(priority + ["remainder"]):
        y = legend_y + index * 24
        rgb = COLORS.get(name, (100, 100, 100))
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(overlay, (legend_x, y - 12), (legend_x + 14, y + 2), bgr, -1)
        cv2.putText(overlay, name, (legend_x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "semantic_overlay.png"), overlay)

    metadata = {}
    for name, mask in masks.items():
        write_mask(out_dir / f"{name}_mask.png", mask)
        ys, xs = np.where(mask > 0)
        metadata[name] = {
            "pixels": int(np.count_nonzero(mask)),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "file": f"{name}_mask.png",
        }

    (out_dir / "semantic_masks.json").write_text(json.dumps(metadata, indent=2))
    print(out_dir / "semantic_overlay.png")
    print(out_dir / "semantic_masks.json")


if __name__ == "__main__":
    main()
