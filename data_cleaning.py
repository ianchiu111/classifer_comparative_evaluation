
import gc
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent
# Clustering_DATA_DIR = BASE_DIR / "data"
# os.makedirs(Clustering_DATA_DIR, exist_ok=True)

@dataclass()
class ColumnConfig:
    time_cols: List[str] = field(default_factory=list)
    int_cols: List[str] = field(default_factory=list)
    float_cols: List[str] = field(default_factory=list)
    string_cols: List[str] = field(default_factory=list)
    bool_cols: List[str] = field(default_factory=list)
    category_cols: List[str] = field(default_factory=list)

class DataCleaning:

    def __init__(self, columns: Optional[ColumnConfig]):
        self.columns = columns

    # ---------- utilities ----------
    @staticmethod
    def _ensure_cols_exist(df: pd.DataFrame, cols: list[str]) -> list[str]:
        """
        Return only columns that exist in df (avoid KeyError).
        """
    
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _normalize_na_strings(s: pd.Series) -> pd.Series:
        """
        Turn common NA tokens into actual NA.
        """

        if not isinstance(s.dtype, pd.StringDtype) and s.dtype != "object":
            s = s.astype("string")

        s = s.str.strip()
        na_tokens = {"", "NA", "N/A", "NULL", "NONE", "NAN", "-", "."}
        return s.mask(s.str.upper().isin(na_tokens), pd.NA)

    # ---------- data type cleaning ----------
    def clean_int_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No integer columns to clean.")
            return df

        print("Cleaning integer columns...")

        out = df.copy()
        for c in cols:
            s = self._normalize_na_strings(out[c].astype("string"))
            s = s.str.replace(",", "", regex=False).str.replace("-", "", regex=False)
            s = s.str.replace(r"\s+", "", regex=True)

            out[c] = pd.to_numeric(s, errors="coerce").astype("Int64")
        return out

    def clean_float_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No float columns to clean.")
            return df

        print("Cleaning float columns...")

        out = df.copy()
        for c in cols:
            s = self._normalize_na_strings(out[c].astype("string"))
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace(r"\s+", "", regex=True)

            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        return out

    def clean_string_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No string columns to clean.")
            return df

        print("Cleaning string columns...")
        out = df.copy()
        for c in cols:
            s = self._normalize_na_strings(out[c].astype("string"))
            out[c] = s
        return out

    def clean_bool_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Convert common boolean representations to pandas 'boolean' dtype:
        true/false, t/f, yes/no, y/n, 1/0.
        """
        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No boolean columns to clean.")
            return df

        print("Cleaning boolean columns...")

        true_set = {"TRUE", "T", "YES", "Y", "1"}
        false_set = {"FALSE", "F", "NO", "N", "0"}

        out = df.copy()
        for c in cols:
            s = out[c]

            # numeric 0/1
            if pd.api.types.is_numeric_dtype(s):
                out[c] = s.map(lambda x: True if x == 1 else False if x == 0 else pd.NA).astype("boolean")
                continue

            # strings
            s = self._normalize_na_strings(out[c].astype("string"))

            mapped = s.str.strip().str.upper().map(lambda x: True if x in true_set else False if x in false_set else pd.NA)
            out[c] = mapped.astype("boolean")

        return out

    def clean_time_cols(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:

        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No time columns to clean.")
            return df

        print("Cleaning time columns...")

        out = df.copy()
        for c in cols:
            out[c] = (
                pd.to_datetime(out[c], errors='coerce')
            )

        return out
    
    def clean_category_cols(
        self,
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:

        """
        Convert special columns into pandas category dtype.
        """

        cols = self._ensure_cols_exist(df, cols)
        if not cols:
            print("No category columns to clean.")
            return df

        print("Cleaning category columns...")

        out = df.copy()
        for c in cols:
            out[c] = out[c].astype("category")

        return out


    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dtype cleaning.
        """
        
        all_cols = (
            self.columns.string_cols 
            + self.columns.time_cols 
            + self.columns.int_cols 
            + self.columns.float_cols 
            + self.columns.bool_cols
            + self.columns.category_cols
        )

        target_cols = [
            col for col in all_cols
            if isinstance(col, str) and col.strip() and col in data.columns
        ]

        copy_data = data.copy()[target_cols]

        copy_data = self.clean_time_cols(copy_data, self.columns.time_cols)
        copy_data = self.clean_int_cols(copy_data, self.columns.int_cols)
        copy_data = self.clean_float_cols(copy_data, self.columns.float_cols)
        copy_data = self.clean_bool_cols(copy_data, self.columns.bool_cols)
        copy_data = self.clean_category_cols(copy_data, self.columns.category_cols)
        cleaned_data = self.clean_string_cols(copy_data, self.columns.string_cols)


        # close useless dataframes by garbage collector
        del data, copy_data
        gc.collect()
        return cleaned_data
