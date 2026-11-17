"""Robust treatment ellipses and shared limits for Fig. 3a."""
from __future__ import annotations
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


def add_robust_coverage_ellipse(
    ax: Axes,
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    color: str,
    required_points: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Draw the 75% ellipse and return the 85% ellipse bounds for the axis limits."""

    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 5:
        return None

    median = np.median(points, axis=0)
    mad = 1.4826 * np.median(np.abs(points - median), axis=0)
    fallback = np.std(points, axis=0, ddof=1)
    scale = np.where(mad > 1e-10, mad, np.maximum(fallback, 1e-10))
    robust_radius2 = np.sum(((points - median) / scale) ** 2, axis=1)
    core = points[robust_radius2 <= np.quantile(robust_radius2, 0.90)]

    center = core.mean(axis=0)
    covariance = np.cov(core, rowvar=False)
    ridge = max(float(np.trace(covariance)), 1.0) * 1e-10
    covariance = covariance + np.eye(2) * ridge
    inverse_covariance = np.linalg.pinv(covariance)
    delta = points - center
    distance2 = np.einsum("ni,ij,nj->n", delta, inverse_covariance, delta)
    radius2, limit_radius2 = np.quantile(distance2, [0.75, 0.85])
    required = np.asarray(required_points, dtype=float)
    required = required[np.isfinite(required).all(axis=1)]
    if len(required):
        required_delta = required - center
        required_distance2 = np.einsum("ni,ij,nj->n", required_delta, inverse_covariance, required_delta)
        # Expand both contours to contain every time-point centroid.
        minimum_radius2 = float(required_distance2.max()) * 1.03
        radius2 = max(radius2, minimum_radius2)
        limit_radius2 = max(limit_radius2, minimum_radius2)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    width, height = 2.0 * np.sqrt(eigenvalues * radius2)
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    ellipse = Ellipse(
        xy=center,
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=mpl.colors.to_rgba(color, 0.055),
        edgecolor=color,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(ellipse)
    width, height = 2.0 * np.sqrt(eigenvalues * limit_radius2)
    theta = np.radians(angle)
    x_radius = 0.5 * np.sqrt((width * np.cos(theta)) ** 2 + (height * np.sin(theta)) ** 2)
    y_radius = 0.5 * np.sqrt((width * np.sin(theta)) ** 2 + (height * np.cos(theta)) ** 2)
    return (
        float(center[0] - x_radius), float(center[0] + x_radius),
        float(center[1] - y_radius), float(center[1] + y_radius),
    )



def calculate_longitudinal_ellipse_limits(
    pc: pd.DataFrame,
    ellipse_bounds: list[tuple[float, float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculate shared plot limits that contain data and treatment ellipses."""

    xmin, xmax = np.quantile(pc["DC1"], [0.005, 0.995])
    ymin, ymax = np.quantile(pc["DC2"], [0.005, 0.995])
    if ellipse_bounds:
        xmin = min(xmin, min(bounds[0] for bounds in ellipse_bounds))
        xmax = max(xmax, max(bounds[1] for bounds in ellipse_bounds))
        ymin = min(ymin, min(bounds[2] for bounds in ellipse_bounds))
        ymax = max(ymax, max(bounds[3] for bounds in ellipse_bounds))
    xpad, ypad = 0.035 * (xmax - xmin), 0.035 * (ymax - ymin)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)
