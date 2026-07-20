from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def component_solidity(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(contour) for contour in contours)
    hull_area = sum(cv2.contourArea(cv2.convexHull(contour)) for contour in contours)
    return contour_area / hull_area if hull_area else 0.0


def split_centerline_and_dense_regions(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    centerline_mask = np.zeros_like(mask, dtype=bool)
    dense_contours: list[np.ndarray] = []

    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < 20:
            continue

        component = labels == label
        fill_ratio = area / float(width * height)
        solidity = component_solidity(component.astype(np.uint8))

        # Filled pupils/eyes/eyebrows become bad medial-axis branches if skeletonized.
        # Treat compact, dense ink islands as outline paths instead.
        is_dense_feature = (
            area >= 70
            and width <= 120
            and height <= 65
            and fill_ratio >= 0.25
            and solidity >= 0.25
        )

        if is_dense_feature:
            comp_uint8 = component.astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if cv2.arcLength(contour, True) >= 8:
                    dense_contours.append(contour)
        else:
            centerline_mask |= component

    return centerline_mask, dense_contours


def skeleton_paths(mask: np.ndarray) -> list[np.ndarray]:
    skeleton = skeletonize(mask).astype(np.uint8)
    ys, xs = np.where(skeleton > 0)
    pixels = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = point
        return [(x + dx, y + dy) for dx, dy in NEIGHBORS_8 if (x + dx, y + dy) in pixels]

    degree = {point: len(neighbors(point)) for point in pixels}
    nodes = {point for point, point_degree in degree.items() if point_degree != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((a, b)))

    for node in list(nodes):
        for neighbor in neighbors(node):
            key = edge_key(node, neighbor)
            if key in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(key)
            previous, current = node, neighbor
            while current not in nodes:
                candidates = [candidate for candidate in neighbors(current) if candidate != previous]
                if not candidates:
                    break
                next_point = candidates[0]
                key = edge_key(current, next_point)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(next_point)
                previous, current = current, next_point
            if len(path) >= 8:
                paths.append(path)

    for point in list(pixels):
        point_neighbors = neighbors(point)
        if not point_neighbors or all(edge_key(point, neighbor) in visited_edges for neighbor in point_neighbors):
            continue
        path = [point]
        previous = None
        current = point
        for _ in range(len(pixels)):
            candidates = [
                candidate
                for candidate in neighbors(current)
                if candidate != previous and edge_key(current, candidate) not in visited_edges
            ]
            if not candidates:
                break
            next_point = candidates[0]
            visited_edges.add(edge_key(current, next_point))
            path.append(next_point)
            previous, current = current, next_point
            if current == point:
                break
        if len(path) >= 12:
            paths.append(path)

    simplified_paths: list[np.ndarray] = []
    for path in paths:
        path_array = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(path_array, 1.8, False).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(simplified.astype(float), axis=0), axis=1)))
        if length >= 8:
            simplified_paths.append(simplified)

    return simplified_paths


def read_points(path: Path, max_points: int = 60, min_distance: int = 13) -> list[tuple[int, int, int]]:
    if not path or not path.exists():
        return []

    raw_points: list[tuple[int, int, int]] = []
    lines = path.read_text().strip().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            x, y, point_type = map(float, parts[:3])
            raw_points.append((int(round(x)), int(round(y)), int(round(point_type))))

    filtered: list[tuple[int, int, int]] = []
    for point in raw_points:
        x, y, _ = point
        if all((x - other_x) ** 2 + (y - other_y) ** 2 >= min_distance ** 2 for other_x, other_y, _ in filtered):
            filtered.append(point)
        if len(filtered) >= max_points:
            break
    return filtered


def path_to_svg_d(points: np.ndarray, closed: bool = False) -> str:
    coords = points.reshape(-1, 2).tolist()
    command = "M " + " L ".join(f"{int(x)} {int(y)}" for x, y in coords)
    return command + (" Z" if closed else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--points")
    parser.add_argument("--png", required=True)
    parser.add_argument("--svg", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    points_path = Path(args.points) if args.points else None
    png_path = Path(args.png)
    svg_path = Path(args.svg)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {input_path}")

    height, width = image.shape
    mask = image < 210
    centerline_mask, dense_contours = split_centerline_and_dense_regions(mask)
    centerline_paths = skeleton_paths(centerline_mask)
    points = read_points(points_path) if points_path else []

    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    original_ink = mask.astype(np.uint8) * 255
    fat_ink = cv2.dilate(original_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    canvas[fat_ink > 0] = (172, 176, 172)
    canvas[original_ink > 0] = (132, 136, 132)

    orange = (0, 150, 255)
    for path in centerline_paths:
        cv2.polylines(canvas, [path.reshape(-1, 1, 2)], False, orange, 3, cv2.LINE_AA)
    for contour in dense_contours:
        simplified = cv2.approxPolyDP(contour, 1.5, True)
        cv2.polylines(canvas, [simplified], True, orange, 3, cv2.LINE_AA)

    blue = (205, 20, 0)
    for index, (x, y, _) in enumerate(points):
        cv2.putText(canvas, str(index), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.72, blue, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 2, blue, -1, cv2.LINE_AA)

    cv2.imwrite(str(png_path), canvas)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]
    for path in centerline_paths:
        svg_parts.append(
            f'<path d="{path_to_svg_d(path)}" fill="none" stroke="#ff9800" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for contour in dense_contours:
        simplified = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        svg_parts.append(
            f'<path d="{path_to_svg_d(simplified, closed=True)}" fill="none" stroke="#ff9800" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    svg_parts.append("</svg>")
    svg_path.write_text("\n".join(svg_parts))

    print(f"centerline_paths={len(centerline_paths)} dense_outline_paths={len(dense_contours)} points={len(points)}")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
