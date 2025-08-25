"""Calculation and filtering utilities for the Streamlit dashboard.

This module centralizes business logic for computing KPIs and applying
filters to data frames so that the UI (app.py) stays clean and declarative.

It provides both:
- A class-based engine (FilterEngine) for scalable, multi-field filtering.
- Backward-compatible functional helpers used by earlier versions of the app.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import json
import ast
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FilterConfig:
	"""Configuration for a single filter.

	kind:
		- 'categorical' -> expects a list of values
		- 'numeric' -> expects a (min, max) range (inclusive)
		- 'date' -> expects a (start_date, end_date) range (inclusive)
	"""

	column: str
	kind: str  # 'categorical' | 'numeric' | 'date'


class FilterEngine:
	"""Engine to compute options and apply multiple filters in a robust way.

	Supported default filters:
	  - asin (categorical)
	  - region (categorical)
	  - sentiment (categorical)
	  - primary_category (categorical)
	  - urgency_score (numeric)
	  - review_date (date)
	"""

	DEFAULT_FILTERS: Tuple[FilterConfig, ...] = (
		FilterConfig("asin", "categorical"),
		FilterConfig("region", "categorical"),
		FilterConfig("sentiment", "categorical"),
		FilterConfig("primary_category", "categorical"),
		FilterConfig("urgency_score", "numeric"),
		FilterConfig("review_date", "date"),
	)

	def __init__(self, df: Optional[pd.DataFrame], filters: Optional[Iterable[FilterConfig]] = None) -> None:
		self.df = df if df is not None else pd.DataFrame()
		self.filters: Tuple[FilterConfig, ...] = tuple(filters) if filters is not None else self.DEFAULT_FILTERS
		self._normalized_df = self._normalize_df(self.df)

	@staticmethod
	def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
		"""Normalize dtypes needed for filtering (e.g., parse dates), safely.

		- Ensures review_date is datetime if present.
		- Leaves other columns untouched aside from no-op copies.
		"""
		if df is None or df.empty:
			return df

		normalized = df.copy()
		if "review_date" in normalized.columns:
			# Coerce to timezone-aware UTC for robust comparisons; keep NaT where not parseable
			normalized["review_date"] = pd.to_datetime(normalized["review_date"], errors="coerce", utc=True)
		return normalized

	def get_filter_options(self) -> Dict[str, Any]:
		"""Return available options/ranges for configured filters.

		Returns a dict with keys per filter column:
		  - categorical: list[str]
		  - numeric: {"min": float, "max": float}
		  - date: {"min": pd.Timestamp | None, "max": pd.Timestamp | None}
		Missing columns yield empty lists/dicts accordingly.
		"""
		out: Dict[str, Any] = {}
		if self._normalized_df is None or self._normalized_df.empty:
			# Initialize empty shapes for all configured filters
			for f in self.filters:
				if f.kind == "categorical":
					out[f.column] = []
				elif f.kind in ("numeric", "date"):
					out[f.column] = {"min": None, "max": None}
			return out

		for f in self.filters:
			if f.column not in self._normalized_df.columns:
				if f.kind == "categorical":
					out[f.column] = []
				else:
					out[f.column] = {"min": None, "max": None}
				continue

			col = self._normalized_df[f.column]
			if f.kind == "categorical":
				vals = (
					col.dropna().astype(str).unique().tolist()
				)
				out[f.column] = sorted(vals)
			elif f.kind == "numeric":
				# Coerce to numeric and compute min/max if any valid values
				numeric = pd.to_numeric(col, errors="coerce")
				if numeric.dropna().empty:
					out[f.column] = {"min": None, "max": None}
				else:
					out[f.column] = {"min": float(numeric.min()), "max": float(numeric.max())}
			elif f.kind == "date":
				dates = pd.to_datetime(col, errors="coerce", utc=True)
				if dates.dropna().empty:
					out[f.column] = {"min": None, "max": None}
				else:
					out[f.column] = {"min": dates.min(), "max": dates.max()}

		return out

	def apply(self, selections: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
		"""Apply filters in 'selections' and return the filtered DataFrame.

		selections structure by kind:
		  - categorical: list[str]
		  - numeric: (min, max)
		  - date: (start, end) where items can be datetime/date/str/Timestamp
		"""
		if self._normalized_df is None or self._normalized_df.empty:
			return self._normalized_df

		if not selections:
			return self._normalized_df

		df = self._normalized_df
		mask = pd.Series(True, index=df.index)

		def _coerce_list(values: Any) -> List[str]:
			if values is None:
				return []
			if isinstance(values, (list, tuple, set)):
				return [str(v) for v in values if v is not None]
			return [str(values)]

		for f in self.filters:
			if f.column not in df.columns:
				continue
			sel = selections.get(f.column) if selections else None

			if f.kind == "categorical":
				selected_values = _coerce_list(sel)
				if selected_values:
					mask &= df[f.column].astype(str).isin(selected_values)

			elif f.kind == "numeric":
				if sel is None:
					continue
				try:
					min_v, max_v = sel  # expects iterable of two
				except Exception:
					continue
				series = pd.to_numeric(df[f.column], errors="coerce")
				if min_v is not None:
					mask &= series >= float(min_v)
				if max_v is not None:
					mask &= series <= float(max_v)

			elif f.kind == "date":
				if sel is None:
					continue
				try:
					start_v, end_v = sel
				except Exception:
					continue
				series = pd.to_datetime(df[f.column], errors="coerce", utc=True)

				def to_ts(x: Any) -> Optional[pd.Timestamp]:
					if x is None:
						return None
					try:
						# Convert to UTC tz-aware timestamp to match series
						return pd.to_datetime(x, utc=True)
					except Exception:
						return None

				start_ts, end_ts = to_ts(start_v), to_ts(end_v)
				if start_ts is not None:
					mask &= series >= start_ts
				if end_ts is not None:
					mask &= series <= end_ts

		return df[mask]


def get_filter_options(df: pd.DataFrame) -> Dict[str, Any]:
	"""Backward-compatible wrapper returning filter options.

	Includes options for asin, region, sentiment, primary_category and ranges for
	urgency_score and review_date when present.
	"""
	engine = FilterEngine(df)
	return engine.get_filter_options()


def apply_filters(
	df: pd.DataFrame,
	asin_values: Optional[List[str]] | None = None,
	region_values: Optional[List[str]] | None = None,
	sentiment_values: Optional[List[str]] | None = None,
	primary_category_values: Optional[List[str]] | None = None,
	urgency_score_range: Optional[Tuple[Optional[float], Optional[float]]] | None = None,
	review_date_range: Optional[Tuple[Any, Any]] | None = None,
) -> pd.DataFrame:
	"""Backward-compatible wrapper to apply filters by individual params.

	New code is encouraged to build a selections dict and use FilterEngine.apply().
	"""
	engine = FilterEngine(df)
	selections: Dict[str, Any] = {}
	if asin_values:
		selections["asin"] = asin_values
	if region_values:
		selections["region"] = region_values
	if sentiment_values:
		selections["sentiment"] = sentiment_values
	if primary_category_values:
		selections["primary_category"] = primary_category_values
	if urgency_score_range:
		selections["urgency_score"] = urgency_score_range
	if review_date_range:
		selections["review_date"] = review_date_range
	return engine.apply(selections)


class KpiEngine:
	"""Compute KPI metrics over time windows using the dataset.

	Defaults:
	  - Time window: last 30 days based on max(review_date) present in the data.
	  - Sentiment mapping: {'Positive': 10, 'Neutral': 5, 'Mixed': 3, 'Negative': 1}.
	  - Urgent issue definition (current implementation):
		  * urgent if urgency_score >= urgent_threshold (default 3)
			OR issue_tags contains at least one tag.
		Breakdown (modifiable):
		  * critical: urgency_score >= 5
		  * high: 4 <= urgency_score < 5

	Note: All date comparisons are done in UTC and inclusive of endpoints.
	"""

	DEFAULT_MAPPING = {
		"Positive": 10.0,
		"Neutral": 5.0,
		"Mixed": 3.0,
		"Negative": 1.0,
	}

	def __init__(self, df: Optional[pd.DataFrame]) -> None:
		self.df = df if df is not None else pd.DataFrame()
		# Reuse date normalization from FilterEngine
		self.df = FilterEngine._normalize_df(self.df)

	def _get_period_masks(self, days: int = 30) -> Tuple[pd.Series, Optional[pd.Series]]:
		"""Return boolean masks for current and previous periods.

		If review_date column is missing or empty, current mask is all True and
		previous mask is None.
		"""
		if self.df is None or self.df.empty or "review_date" not in self.df.columns:
			return pd.Series(True, index=self.df.index), None

		dates = pd.to_datetime(self.df["review_date"], errors="coerce", utc=True)
		if dates.dropna().empty:
			return pd.Series(True, index=self.df.index), None

		end_current = dates.max()  # tz-aware UTC
		# Normalize to date boundaries (inclusive)
		end_current = end_current.floor("D")
		start_current = end_current - pd.Timedelta(days=days - 1)

		# Previous window is the immediately preceding block of equal length
		end_prev = start_current - pd.Timedelta(days=1)
		start_prev = end_prev - pd.Timedelta(days=days - 1)

		cur_mask = (dates >= start_current) & (dates <= end_current)
		prev_mask = (dates >= start_prev) & (dates <= end_prev)
		return cur_mask, prev_mask

	def sentiment_score(
		self,
		days: int = 30,
		mapping: Optional[Mapping[str, float]] = None,
	) -> Dict[str, Optional[float]]:
		if self.df is None or self.df.empty or "sentiment" not in self.df.columns:
			return {"score": None, "delta": None}

		cur_mask, prev_mask = self._get_period_masks(days)
		use_map = dict(self.DEFAULT_MAPPING)
		if mapping:
			use_map.update(mapping)

		def to_score(s: Any) -> Optional[float]:
			if s is None:
				return None
			return use_map.get(str(s), None)

		scores = self.df["sentiment"].map(to_score)
		cur_scores = scores[cur_mask].dropna()
		cur = float(cur_scores.mean()) if not cur_scores.empty else None

		prev = None
		if prev_mask is not None:
			prev_scores = scores[prev_mask].dropna()
			prev = float(prev_scores.mean()) if not prev_scores.empty else None

		delta = None
		if cur is not None and prev is not None:
			delta = cur - prev

		return {"score": cur, "delta": delta}

	def review_volume(self, days: int = 30) -> Dict[str, Optional[float]]:
		if self.df is None or self.df.empty:
			return {"count": 0, "delta_pct": None}

		if "review_date" not in self.df.columns:
			# No time dimension; return full size and no delta
			return {"count": int(len(self.df)), "delta_pct": None}

		cur_mask, prev_mask = self._get_period_masks(days)
		cur_count = int(cur_mask.sum())
		prev_count = int(prev_mask.sum()) if prev_mask is not None else 0

		delta_pct = None
		if prev_count > 0:
			delta_pct = ((cur_count - prev_count) / prev_count) * 100.0

		return {"count": cur_count, "delta_pct": delta_pct}

	@staticmethod
	def _parse_issue_tags(val: Any) -> List[str]:
		"""Parse issue_tags like '{"wrong shade","wrong finish"}' into a list.

		Handles None/empty and tolerates stray spaces/quotes/braces.
		"""
		if val is None:
			return []
		s = str(val).strip()
		if not s or s == "{}" or s == "{ }":
			return []
		# Remove surrounding braces if present
		if s.startswith("{") and s.endswith("}"):
			s = s[1:-1]
		# Split by comma and strip quotes/space
		parts: List[str] = []
		for p in s.split(","):
			p = p.strip()
			if not p:
				continue
			# Strip both double and single quotes from each part
			p = p.strip("\"'")
			if p:
				parts.append(p)
		return parts

	def urgent_issues(self, days: int = 30, urgent_threshold: int = 3) -> Dict[str, int]:
		"""Count urgent issues in the current window.

		Current heuristic:
		  - urgent if urgency_score >= urgent_threshold OR issue_tags has entries.
		  - breakdown (modifiable in future):
			  * critical: urgency_score >= 5
			  * high: 4 <= urgency_score < 5

		Returns dict with 'total', 'critical', 'high'.
		"""
		if self.df is None or self.df.empty:
			return {"total": 0, "critical": 0, "high": 0}

		cur_mask, _ = self._get_period_masks(days)
		dfc = self.df[cur_mask].copy()

		urgency = pd.to_numeric(dfc.get("urgency_score", pd.Series(index=dfc.index)), errors="coerce")

		# Updated rule (per product requirements):
		# Urgent issues are defined ONLY by urgency_score >= urgent_threshold.
		# Do not include issue_tags presence in the urgent determination.
		urgent_mask = urgency.fillna(0) >= float(urgent_threshold)

		total = int(urgent_mask.sum())

		# Simple breakdown using urgency_score; safe if column missing
		critical = int((urgency >= 5).fillna(False).sum())
		high = int(((urgency >= 4) & (urgency < 5)).fillna(False).sum())

		return {"total": total, "critical": critical, "high": high}

	def team_utilization(self) -> Optional[float]:
		"""Placeholder: no data available. Return None to keep UI blank."""
		return None

	def compute_all(self, days: int = 30) -> Dict[str, Any]:
		"""Compute all KPIs into a single dict."""
		out: Dict[str, Any] = {}
		out["sentiment_score"] = self.sentiment_score(days=days)
		out["review_volume"] = self.review_volume(days=days)
		out["urgent_issues"] = self.urgent_issues(days=days)
		out["team_utilization"] = self.team_utilization()
		return out

	# --- Key Drivers (Positive/Negative) from `key_drivers` column ---

	@staticmethod
	def _parse_key_drivers_cell(v: Any) -> Dict[str, Any]:
		"""Parse a cell from `key_drivers` into a dict.

		Accepts dict, JSON string, or Python-literal string. Returns {} when invalid.
		"""
		if isinstance(v, dict):
			return v
		if v is None or (isinstance(v, float) and pd.isna(v)):
			return {}
		s = str(v).strip()
		if not s:
			return {}
		# Try JSON first
		try:
			obj = json.loads(s)
			return obj if isinstance(obj, dict) else {}
		except Exception:
			# Fallback to Python literal
			try:
				obj = ast.literal_eval(s)
				return obj if isinstance(obj, dict) else {}
			except Exception:
				return {}

	def key_driver_lists(self, days: Optional[int] = None) -> pd.DataFrame:
		"""Return a DataFrame with 'Positive' and 'Negative' list columns.

		For each row of self.df (optionally limited to the current time window when
		`days` is provided), reads key_drivers (dict mapping topic->sentiment label)
		and builds two list columns. When a list would be empty, stores np.nan to
		mirror the ruf.ipynb logic.
		"""
		if self.df is None or self.df.empty or "key_drivers" not in self.df.columns:
			return pd.DataFrame({"Positive": pd.Series(dtype=object), "Negative": pd.Series(dtype=object)})

		# Scope to current period if days is provided and dates are available
		df_scope = self.df
		if days is not None and "review_date" in df_scope.columns:
			cur_mask, _ = self._get_period_masks(days)
			df_scope = df_scope[cur_mask]
			if df_scope.empty:
				return pd.DataFrame({"Positive": pd.Series(dtype=object), "Negative": pd.Series(dtype=object)})

		kd = df_scope["key_drivers"].apply(self._parse_key_drivers_cell)
		pos = kd.apply(lambda d: [k for k, v in d.items() if str(v).strip().lower() == "positive"])  # type: ignore
		neg = kd.apply(lambda d: [k for k, v in d.items() if str(v).strip().lower() == "negative"])  # type: ignore

		# Replace empty lists with np.nan to match notebook semantics
		pos = pos.apply(lambda lst: lst if lst else np.nan)
		neg = neg.apply(lambda lst: lst if lst else np.nan)

		return pd.DataFrame({"Positive": pos, "Negative": neg}, index=df_scope.index)

	def top_key_drivers(self, n: int = 6, days: Optional[int] = None) -> Dict[str, List[Tuple[str, int]]]:
		"""Compute top-N Positive and Negative drivers with counts.

		- Works on the currently attached/filtered DataFrame.
		- Uses explode().value_counts() like ruf.ipynb to rank topics.
		Returns: {"positive": [(topic, count), ...], "negative": [...]} (max N each)
		"""
		lists_df = self.key_driver_lists(days=days)
		if lists_df.empty:
			return {"positive": [], "negative": []}

		pos_counts = pd.Series(dtype=int)
		neg_counts = pd.Series(dtype=int)
		try:
			pos_counts = lists_df["Positive"].dropna().explode().astype(str).str.strip().replace("", np.nan).dropna().value_counts()
		except Exception:
			pass
		try:
			neg_counts = lists_df["Negative"].dropna().explode().astype(str).str.strip().replace("", np.nan).dropna().value_counts()
		except Exception:
			pass

		pos_top = list(pos_counts.head(n).items()) if not pos_counts.empty else []
		neg_top = list(neg_counts.head(n).items()) if not neg_counts.empty else []
		return {"positive": pos_top, "negative": neg_top}

	def topic_drivers(
		self,
		topics: Optional[List[Tuple[str, List[str]]]] = None,
		text_columns: Optional[List[str]] = None,
		mapping: Optional[Mapping[str, float]] = None,
	) -> List[Dict[str, Any]]:
		"""Compute topic drivers from available text columns and issue_tags.

		Returns a list of dicts with keys: title, impact, mentions, pct.
		- impact: difference between topic mean sentiment (0–10) and overall mean.
		- mentions: number of reviews matching topic keywords/tags.
		- pct: mentions / total_filtered_reviews * 100.
		"""
		if self.df is None or self.df.empty:
			return []

		# Determine text sources
		default_text_cols = [
			"review_text", "review", "reviewText", "summary", "title",
		]
		cols = text_columns or [c for c in default_text_cols if c in self.df.columns]

		# Build a lowercase combined text field
		if cols:
			text_series = self.df[cols].astype(str).agg(" ".join, axis=1).str.lower()
		else:
			text_series = pd.Series([""] * len(self.df), index=self.df.index)

		# Include parsed issue_tags as part of the text corpus
		if "issue_tags" in self.df.columns:
			parsed = self.df["issue_tags"].apply(self._parse_issue_tags).apply(lambda lst: " ".join(lst).lower())
			text_series = (text_series.fillna("") + " " + parsed.fillna(""))

		# Map sentiment to 0–10 scale
		use_map = dict(self.DEFAULT_MAPPING)
		if mapping:
			use_map.update(mapping)
		sent_scores = self.df.get("sentiment")
		if sent_scores is None:
			return []
		mapped = sent_scores.map(lambda s: use_map.get(str(s), None))
		overall_mean = float(pd.to_numeric(mapped, errors="coerce").dropna().mean()) if mapped.notna().any() else None
		if overall_mean is None:
			return []

		# Default topic keywords if not provided
		if topics is None:
			topics = [
				("Perfect Color Match", ["color match", "shade match", "wrong shade", "color mismatch"]),
				("Long-Lasting Formula", ["long lasting", "longevity", "wear time", "lasts all day"]),
				("Gentle on Sensitive Skin", ["sensitive skin", "allergic", "allergy", "irritation", "rash"]),
				("Beautiful Packaging", ["packaging", "package", "box", "bottle"]),
				("Great Coverage", ["coverage", "cover", "full coverage"]),
				("Delivery Speed", ["deliver", "delivery", "shipping", "arrived", "late"]),
				("Value for Money", ["value", "price", "worth", "expensive", "cheap"]),
			]

		results: List[Dict[str, Any]] = []
		n = len(self.df)
		for title, kws in topics:
			if not kws:
				continue
			pattern = "|".join([re.escape(k.lower()) for k in kws])
			mask = text_series.str.contains(pattern, case=False, na=False, regex=True)
			mentions = int(mask.sum())
			pct = float(mentions / n * 100.0) if n > 0 else 0.0
			if mentions > 0:
				topic_scores = pd.to_numeric(mapped[mask], errors="coerce").dropna()
				topic_mean = float(topic_scores.mean()) if not topic_scores.empty else overall_mean
			else:
				topic_mean = overall_mean
			impact = round((topic_mean - overall_mean), 1) if topic_mean is not None else 0.0
			results.append({
				"title": title,
				"impact": impact,
				"mentions": mentions,
				"pct": round(pct),
			})

		return results

	def sentiment_trend(
		self,
		freq: str = "ME",
		smoothing_window: int = 0,
		mapping: Optional[Mapping[str, float]] = None,
	) -> pd.DataFrame:
		"""Return a DataFrame with average sentiment score by period and optional smoothing.

		Columns returned: [period, score, smoothed]
		- period: period end timestamp
		- score: average mapped sentiment per period (NaNs dropped)
		- smoothed: rolling mean over 'score' when smoothing_window > 1 else equals 'score'
		"""
		if self.df is None or self.df.empty:
			return pd.DataFrame(columns=["period", "score", "smoothed"])

		if "review_date" not in self.df.columns or "sentiment" not in self.df.columns:
			return pd.DataFrame(columns=["period", "score", "smoothed"])

		dt = pd.to_datetime(self.df["review_date"], errors="coerce", utc=True)
		if dt.dropna().empty:
			return pd.DataFrame(columns=["period", "score", "smoothed"])

		use_map = dict(self.DEFAULT_MAPPING)
		if mapping:
			use_map.update(mapping)

		scores = self.df["sentiment"].map(lambda s: use_map.get(str(s), None))
		d = pd.DataFrame({"date": dt, "score": scores})
		d = d.dropna(subset=["date", "score"]).sort_values("date")
		if d.empty:
			return pd.DataFrame(columns=["period", "score", "smoothed"])

		d = d.set_index("date").resample(freq)["score"].mean().dropna().to_frame()
		d = d.reset_index()
		d.rename(columns={"date": "period"}, inplace=True)

		if smoothing_window and smoothing_window > 1:
			d["smoothed"] = d["score"].rolling(window=int(smoothing_window), min_periods=1).mean()
		else:
			d["smoothed"] = d["score"]

		return d


def compute_kpis(df: pd.DataFrame) -> Dict[str, int]:
	"""Compute KPI values from the (optionally filtered) DataFrame.

	Args:
		df: DataFrame after filters are applied.

	Returns:
		Dict with KPI names and values. For now only 'rows'.
	"""
	if df is None or df.empty:
		return {"rows": 0}

	# Backward-compatible minimal KPI; for full KPI set use KpiEngine.compute_all().
	return {"rows": int(len(df))}
