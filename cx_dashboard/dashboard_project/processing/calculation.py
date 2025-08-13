"""Calculation and filtering utilities for the Streamlit dashboard.

This module centralizes business logic for computing KPIs and applying
filters to data frames so that the UI (app.py) stays clean and declarative.
"""

from typing import Dict, List

import pandas as pd


def get_filter_options(df: pd.DataFrame) -> Dict[str, List[str]]:
	"""Return available filter options from the dataset.

	Currently provides unique ASIN values (non-null), sorted as strings.

	Args:
		df: Input DataFrame.

	Returns:
		Dict of filter_name -> list of choices. Example: {"asin": ["B001", ...]}
	"""
	options: Dict[str, List[str]] = {"asin": []}
	if df is None or df.empty:
		return options

	if "asin" in df.columns:
		values = (
			df["asin"]
			.dropna()
			.astype(str)
			.unique()
			.tolist()
		)
		options["asin"] = sorted(values)

	return options


def apply_filters(df: pd.DataFrame, asin_values: List[str] | None = None) -> pd.DataFrame:
	"""Apply UI filters to the dataset and return the filtered DataFrame.

	Args:
		df: Input DataFrame.
		asin_values: List of selected ASINs to include. If empty/None, no ASIN filter.

	Returns:
		Filtered DataFrame (same instance if no filters applied).
	"""
	if df is None or df.empty:
		return df

	filtered = df

	if asin_values:
		if "asin" in filtered.columns:
			# Ensure comparison is performed on string representations
			filtered = filtered[filtered["asin"].astype(str).isin([str(v) for v in asin_values])]

	return filtered


def compute_kpis(df: pd.DataFrame) -> Dict[str, int]:
	"""Compute KPI values from the (optionally filtered) DataFrame.

	Args:
		df: DataFrame after filters are applied.

	Returns:
		Dict with KPI names and values. For now only 'rows'.
	"""
	if df is None or df.empty:
		return {"rows": 0}

	return {"rows": int(len(df))}

