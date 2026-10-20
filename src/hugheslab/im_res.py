"""
Functions for resolution analysis of images.

"""

from dataclasses import dataclass


import cv2

import numpy as np



@dataclass
class EdgeSpreadFunction:
    distance_px: np.ndarray
    intensity: np.ndarray
    sample_count: np.ndarray

    edge_point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray

    edge_angle_deg: float
    edge_position_px: float


@dataclass
class SquareDetection:
    corners: np.ndarray          # Shape: (4, 2)
    center: tuple[float, float]
    width: float
    height: float
    angle: float
    score: float


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Return corners in this order:
    top-left, top-right, bottom-right, bottom-left.
    """
    points = np.asarray(points, dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )



def find_square(img, size= 10, blur = 5):
    """ Finds a sqaure in a grayscale image. Mostly used for finding the square 
    in a USAF resolution target.

    Arguments:
    img:    ndarray
            2D array containing grayscale image

    Keyword Arguments:
    size:   int
            Expected size of the square in pixels. Default is 10.
    blur:   int
            Size of the Gaussian blur kernel. Default is 5.
    """
    img = cv2.GaussianBlur(img, (blur, blur), 0)


    # Canny thresholds derived from the image median.
    median = float(np.median(img))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(img, lower, upper)

    # Join small gaps in the square border.
    kernel_size = max(3, int(round(size * 0.02)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_size = size * (1.0 - size)
    max_size = size * (1.0 + size)

    best = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect

        if width <= 0 or height <= 0:
            continue

        mean_size = (width + height) / 2.0

        if not min_size <= mean_size <= max_size:
            continue

        # A square should have an aspect ratio near 1.
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 1.25:
            continue

        rect_area = width * height
        fill_ratio = area / rect_area

        #if fill_ratio < min_square_fill:
        #    continue

        box = cv2.boxPoints(rect)
        corners = order_corners(box)

        size_error = abs(mean_size - size) / size
        aspect_error = abs(1.0 - width / height)

        # Lower geometric error and higher fill ratio are preferred.
        score = (
            2.0 * fill_ratio
            - 1.5 * size_error
            - 1.5 * aspect_error
        )

        candidate = SquareDetection(
            corners=corners,
            center=(float(cx), float(cy)),
            width=float(width),
            height=float(height),
            angle=float(angle),
            score=float(score),
        )

        if best is None or candidate.score > best.score:
            best = candidate

    return best



def draw_detection(image, detection):
    result = image.copy()

    polygon = detection.corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(result, [polygon], True, (0, 255, 0), 3)

    center = tuple(round(value) for value in detection.center)
    cv2.circle(result, center, 5, (0, 0, 255), -1)

    return result


def calculate_esf(
    image: np.ndarray,
    edge_start: np.ndarray | tuple[float, float],
    edge_end: np.ndarray | tuple[float, float],
    strip_width_px: float = 20.0,
    end_margin_px: float = 5.0,
    oversampling: int = 8,
    channel: int | None = None,
    min_samples_per_bin: int = 1,
    normal_smoothing_sigma: float = 0.0,
) -> EdgeSpreadFunction:
    """
    Calculate an oversampled edge spread function along a slanted edge.

    Parameters
    ----------
    image:
        Grayscale image or BGR colour image.

    edge_start, edge_end:
        Two points defining the square side, in image coordinates (x, y).

    strip_width_px:
        Width of the sampled strip, perpendicular to the edge.
        For example, 20 uses pixels approximately +/- 10 pixels from the edge.

    end_margin_px:
        Distance excluded from each end of the edge. This helps avoid
        contamination from the square's corners.

    oversampling:
        Number of ESF samples per image pixel. For example, 8 gives a
        nominal ESF bin width of 1/8 pixel.

    channel:
        For a colour image:
          None -> convert to grayscale
          0    -> blue channel
          1    -> green channel
          2    -> red channel

    min_samples_per_bin:
        Remove bins containing fewer than this many image pixels.

    normal_smoothing_sigma:
        Optional Gaussian smoothing of the final ESF, measured in ESF bins.
        Use 0 to disable smoothing.

    Returns
    -------
    EdgeSpreadFunction
    """
    if image is None or image.size == 0:
        raise ValueError("image is empty")

    if strip_width_px <= 0:
        raise ValueError("strip_width_px must be positive")

    if oversampling < 1:
        raise ValueError("oversampling must be at least 1")

    if min_samples_per_bin < 1:
        raise ValueError("min_samples_per_bin must be at least 1")

    p0 = np.asarray(edge_start, dtype=np.float64)
    p1 = np.asarray(edge_end, dtype=np.float64)

    if p0.shape != (2,) or p1.shape != (2,):
        raise ValueError("edge_start and edge_end must each contain x and y")

    edge_vector = p1 - p0
    edge_length = float(np.linalg.norm(edge_vector))

    if edge_length <= 2.0 * end_margin_px:
        raise ValueError(
            "The usable edge length must be greater than twice end_margin_px"
        )

    tangent = edge_vector / edge_length

    # Image coordinates use x to the right and y downward.
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3:
        if channel is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            if channel not in (0, 1, 2):
                raise ValueError("channel must be None, 0, 1, or 2")
            gray = image[..., channel].astype(np.float64)
    else:
        raise ValueError("image must be grayscale or BGR")

    height, width = gray.shape

    half_width = strip_width_px / 2.0

    # Bounding box around the finite sampling strip.
    corners = np.array(
        [
            p0 + normal * half_width,
            p0 - normal * half_width,
            p1 + normal * half_width,
            p1 - normal * half_width,
        ]
    )

    x_min = max(0, int(np.floor(corners[:, 0].min())) - 1)
    x_max = min(width - 1, int(np.ceil(corners[:, 0].max())) + 1)
    y_min = max(0, int(np.floor(corners[:, 1].min())) - 1)
    y_max = min(height - 1, int(np.ceil(corners[:, 1].max())) + 1)

    if x_min > x_max or y_min > y_max:
        raise ValueError("The edge sampling area is outside the image")

    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    pixel_points = np.column_stack((xx.ravel(), yy.ravel()))
    relative = pixel_points - p0

    # Coordinate along the edge.
    along = relative @ tangent

    # Signed coordinate perpendicular to the edge.
    across = relative @ normal

    valid = (
        (along >= end_margin_px)
        & (along <= edge_length - end_margin_px)
        & (np.abs(across) <= half_width)
    )

    distances = across[valid]
    intensities = gray[
        yy.ravel()[valid].astype(np.int64),
        xx.ravel()[valid].astype(np.int64),
    ]

    if distances.size == 0:
        raise ValueError("No pixels lie inside the requested edge strip")

    bin_width = 1.0 / oversampling

    first_bin_edge = np.floor(distances.min() / bin_width) * bin_width
    last_bin_edge = np.ceil(distances.max() / bin_width) * bin_width

    bin_edges = np.arange(
        first_bin_edge,
        last_bin_edge + bin_width * 1.01,
        bin_width,
    )

    bin_indices = np.digitize(distances, bin_edges) - 1
    number_of_bins = len(bin_edges) - 1

    in_range = (bin_indices >= 0) & (bin_indices < number_of_bins)
    bin_indices = bin_indices[in_range]
    intensities = intensities[in_range]

    intensity_sum = np.bincount(
        bin_indices,
        weights=intensities,
        minlength=number_of_bins,
    )

    sample_count = np.bincount(
        bin_indices,
        minlength=number_of_bins,
    )

    esf = np.full(number_of_bins, np.nan, dtype=np.float64)

    populated = sample_count >= min_samples_per_bin
    esf[populated] = intensity_sum[populated] / sample_count[populated]

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Interpolate empty bins only when they lie between populated bins.
    finite = np.isfinite(esf)

    if finite.sum() < 2:
        raise ValueError(
            "Too few populated ESF bins. Increase strip_width_px or reduce "
            "oversampling."
        )

    first_valid = np.flatnonzero(finite)[0]
    last_valid = np.flatnonzero(finite)[-1]

    keep = np.arange(first_valid, last_valid + 1)

    bin_centres = bin_centres[keep]
    esf = esf[keep]
    sample_count = sample_count[keep]

    finite = np.isfinite(esf)
    esf = np.interp(
        bin_centres,
        bin_centres[finite],
        esf[finite],
    )

    if normal_smoothing_sigma > 0:
        esf = cv2.GaussianBlur(
            esf.reshape(1, -1),
            ksize=(0, 0),
            sigmaX=normal_smoothing_sigma,
        ).ravel()

    # Estimate the edge location from the 50% transition point.
    low_level = np.percentile(esf, 10)
    high_level = np.percentile(esf, 90)
    midpoint = 0.5 * (low_level + high_level)

    nearest_midpoint_index = int(np.argmin(np.abs(esf - midpoint)))
    edge_position = float(bin_centres[nearest_midpoint_index])

    # Shift the ESF so that approximately zero is the edge centre.
    bin_centres = bin_centres - edge_position

    edge_angle_deg = float(
        np.degrees(np.arctan2(tangent[1], tangent[0]))
    )

    edge_midpoint = 0.5 * (p0 + p1) + normal * edge_position

    return EdgeSpreadFunction(
        distance_px=bin_centres,
        intensity=esf,
        sample_count=sample_count,
        edge_point=edge_midpoint,
        tangent=tangent,
        normal=normal,
        edge_angle_deg=edge_angle_deg,
        edge_position_px=edge_position,
    )