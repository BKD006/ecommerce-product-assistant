import pandas as pd
import numpy as np
import ast
from typing import List, Dict, Any


class ProductCatalogProcessor:
    """
    Loads Flipkart product catalog CSV
    and converts rows into embedding-ready records.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    # -------------------------------------------------
    # PUBLIC METHOD
    # -------------------------------------------------

    def process(self) -> List[Dict[str, Any]]:
        """
        Returns:
        [
            {
                "id": str,
                "embedding_text": str,
                "metadata": dict
            }
        ]
        """
        df = self._load_data()
        df = self._basic_cleaning(df)
        df = self._normalize_price(df)
        df = self._clean_ratings(df)
        df = self._extract_categories(df)
        df = self._parse_specifications(df)
        df = self._validate_rows(df)
        df = self._build_embedding_text(df)
        df = self._build_metadata(df)

        final_df = df[["pid", "embedding_text", "metadata"]].copy()

        return [
            {
                "id": str(row["pid"]),
                "embedding_text": row["embedding_text"],
                "metadata": row["metadata"]
            }
            for _, row in final_df.iterrows()
        ]

    # -------------------------------------------------
    # INTERNAL STEPS
    # -------------------------------------------------
    def _validate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation before building embedding text.
        Removes low-quality or invalid rows.
        """

        # Remove missing required fields
        df = df[df["pid"].notna()]
        df = df[df["price"].notna()]
        df = df[df["main_category"].notna()]

        # Remove invalid prices
        df = df[df["price"] > 0]

        # Remove too-short product names
        df = df[df["product_name"].str.len() > 3]

        # Remove too-short descriptions
        df = df[df["description"].str.len() > 20]

        # Remove extremely long descriptions (optional safety)
        df = df[df["description"].str.len() < 5000]

        return df
    
    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["product_name"].notna()]
        df = df[df["retail_price"].notna() | df["discounted_price"].notna()]
        df = df[df["description"].notna()]
        df = df.drop_duplicates(subset="pid")

        columns_to_drop = [
            "uniq_id",
            "crawl_timestamp",
            "product_url",
            "image",
            "is_FK_Advantage_product"
        ]

        df = df.drop(columns=columns_to_drop, errors="ignore")
        return df

    def _normalize_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = df["discounted_price"].fillna(df["retail_price"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["price"].notna()]
        return df

    def _clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        df["overall_rating"] = df["overall_rating"].replace(
            "No rating available", np.nan
        )
        df["overall_rating"] = pd.to_numeric(
            df["overall_rating"], errors="coerce"
        )
        return df

    # -------------------------------------------------
    # CATEGORY EXTRACTION
    # -------------------------------------------------

    def _extract_categories(self, df: pd.DataFrame) -> pd.DataFrame:

        def extract(cat_string):
            if pd.isna(cat_string):
                return None, None

            try:
                cat_list = ast.literal_eval(cat_string)
                full_path = cat_list[0]
                parts = full_path.split(" >> ")

                main_category = parts[0].strip()
                sub_category = parts[1].strip() if len(parts) > 1 else None

                return main_category, sub_category

            except Exception:
                return None, None

        df[["main_category", "sub_category"]] = df[
            "product_category_tree"
        ].apply(lambda x: pd.Series(extract(x)))

        return df

    # -------------------------------------------------
    # SPEC PARSING
    # -------------------------------------------------

    def _parse_specifications(self, df: pd.DataFrame) -> pd.DataFrame:

        def parse_specifications(spec_string):
            if pd.isna(spec_string):
                return {}

            try:
                cleaned = spec_string.replace("=>", ":")
                parsed = ast.literal_eval(cleaned)
                specs = parsed.get("product_specification", [])

                spec_dict = {}

                for item in specs:
                    key = item.get("key")
                    value = item.get("value")

                    if key and value:
                        spec_dict[key] = value

                return spec_dict

            except Exception:
                return {}

        df["parsed_specs"] = df["product_specifications"].apply(
            parse_specifications
        )

        return df

    # -------------------------------------------------
    # BUILD EMBEDDING TEXT
    # -------------------------------------------------

    def _build_embedding_text(self, df: pd.DataFrame) -> pd.DataFrame:

        def build_text(row):
            specs_text = ""

            if isinstance(row["parsed_specs"], dict):
                for k, v in row["parsed_specs"].items():
                    specs_text += f"{k}: {v}\n"

            text = f"""
            Product: {row['product_name']}
            Brand: {row.get('brand', '')}
            Category: {row.get('main_category', '')}
            Subcategory: {row.get('sub_category', '')}
            Price: {row.get('price', '')}

            Description:
            {row.get('description', '')}

            Specifications:
            {specs_text}
            """

            return text.strip()

        df["embedding_text"] = df.apply(build_text, axis=1)
        return df

    # -------------------------------------------------
    # BUILD METADATA
    # -------------------------------------------------

    def _build_metadata(self, df: pd.DataFrame) -> pd.DataFrame:

        def build_metadata(row):

            metadata = {
                "product_id": str(row["pid"]),
                "category": row.get("main_category"),
                "sub_category": row.get("sub_category"),
                "brand": row.get("brand"),
                "price": float(row.get("price")),
            }

            # Add rating only if exists
            if not pd.isna(row["overall_rating"]):
                metadata["rating"] = float(row["overall_rating"])

            # 🔥 REMOVE ANY NULL VALUES (CRITICAL)
            metadata = {
                k: v for k, v in metadata.items()
                if v is not None and not pd.isna(v)
            }

            return metadata

        df["metadata"] = df.apply(build_metadata, axis=1)
        return df
