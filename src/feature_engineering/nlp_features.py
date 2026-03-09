"""
NLP Feature Engineering for Healthcare Intelligence System.

Extracts TF-IDF features, medical entities, negation patterns, severity
scores, and BioClinicalBERT embeddings from clinical notes.
"""

import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    StopWordsRemover,
    Tokenizer,
)


class NLPFeatureEngineer:
    """Engineers features from unstructured clinical text using PySpark."""

    # ----- medical dictionaries for entity / severity extraction ----- #
    _MEDICATION_PATTERNS = (
        r"\b(?:aspirin|metformin|lisinopril|atorvastatin|amlodipine|"
        r"omeprazole|metoprolol|losartan|albuterol|gabapentin|"
        r"hydrochlorothiazide|acetaminophen|ibuprofen|amoxicillin|"
        r"prednisone|insulin|warfarin|heparin|clopidogrel|furosemide|"
        r"levothyroxine|pantoprazole|sertraline|fluoxetine|"
        r"ciprofloxacin|azithromycin|doxycycline|morphine|fentanyl|"
        r"vancomycin|ceftriaxone)\b"
    )
    _DIAGNOSIS_PATTERNS = (
        r"\b(?:diabetes|hypertension|pneumonia|sepsis|COPD|asthma|"
        r"heart failure|myocardial infarction|stroke|anemia|"
        r"cirrhosis|hepatitis|renal failure|cancer|tumor|malignancy|"
        r"infection|fracture|embolism|thrombosis|fibrillation|"
        r"cardiomyopathy|pancreatitis|meningitis|encephalitis)\b"
    )
    _PROCEDURE_PATTERNS = (
        r"\b(?:intubation|ventilation|catheterization|biopsy|"
        r"dialysis|transfusion|surgery|resection|endoscopy|"
        r"colonoscopy|bronchoscopy|angiography|echocardiogram|"
        r"MRI|CT scan|X-ray|ultrasound|EKG|ECG|lumbar puncture|"
        r"thoracentesis|paracentesis|tracheostomy)\b"
    )
    _ANATOMY_PATTERNS = (
        r"\b(?:lung|liver|kidney|heart|brain|spleen|pancreas|"
        r"stomach|colon|esophagus|bladder|prostate|thyroid|"
        r"adrenal|pituitary|femur|tibia|spine|vertebra|"
        r"aorta|vein|artery|bronchus|trachea|pleura)\b"
    )

    _CRITICAL_KEYWORDS = [
        "critical", "emergent", "acute", "severe", "unstable",
        "deteriorating", "shock", "arrest", "unresponsive", "intubated",
        "septic", "hemorrhage", "code blue", "stat", "life-threatening",
        "respiratory failure", "cardiac arrest", "comatose",
    ]
    _BENIGN_KEYWORDS = [
        "stable", "improving", "resolved", "normal", "unremarkable",
        "benign", "routine", "discharged", "ambulatory", "alert",
        "oriented", "well-appearing", "afebrile", "no complaints",
    ]

    _NEGATION_CUES = [
        "no ", "not ", "denies ", "denied ", "negative for ",
        "without ", "absence of ", "does not have ", "no evidence of ",
        "rules out ", "ruled out ", "free of ",
    ]

    # ------------------------------------------------------------------ #
    #  TF-IDF Pipeline
    # ------------------------------------------------------------------ #
    def extract_tfidf_features(
        self,
        spark: SparkSession,
        notes_df: DataFrame,
        text_col: str = "clinical_notes",
        max_features: int = 5000,
    ) -> DataFrame:
        """Build a TF-IDF feature vector from clinical notes.

        Pipeline: Tokenizer -> StopWordsRemover -> HashingTF -> IDF.

        Returns DataFrame with added column ``tfidf_features``.
        """
        tokenizer = Tokenizer(inputCol=text_col, outputCol="_tokens")
        remover = StopWordsRemover(inputCol="_tokens", outputCol="_filtered")
        hashing_tf = HashingTF(
            inputCol="_filtered",
            outputCol="_raw_tf",
            numFeatures=max_features,
        )
        idf = IDF(inputCol="_raw_tf", outputCol="tfidf_features")

        pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
        model = pipeline.fit(notes_df)
        result = model.transform(notes_df)

        # Drop intermediate columns
        result = result.drop("_tokens", "_filtered", "_raw_tf")
        return result

    # ------------------------------------------------------------------ #
    #  Regex-based NER
    # ------------------------------------------------------------------ #
    def extract_medical_entities(
        self,
        notes_df: DataFrame,
        text_col: str = "clinical_notes",
    ) -> DataFrame:
        """Extract counts of medical entities via regex UDFs.

        Adds columns: medication_count, diagnosis_count, procedure_count,
                      anatomy_count.
        """
        med_pat = self._MEDICATION_PATTERNS
        diag_pat = self._DIAGNOSIS_PATTERNS
        proc_pat = self._PROCEDURE_PATTERNS
        anat_pat = self._ANATOMY_PATTERNS

        @F.udf(IntegerType())
        def _count_medications(text):
            if text is None:
                return 0
            return len(re.findall(med_pat, text, re.IGNORECASE))

        @F.udf(IntegerType())
        def _count_diagnoses(text):
            if text is None:
                return 0
            return len(re.findall(diag_pat, text, re.IGNORECASE))

        @F.udf(IntegerType())
        def _count_procedures(text):
            if text is None:
                return 0
            return len(re.findall(proc_pat, text, re.IGNORECASE))

        @F.udf(IntegerType())
        def _count_anatomy(text):
            if text is None:
                return 0
            return len(re.findall(anat_pat, text, re.IGNORECASE))

        df = notes_df
        df = df.withColumn("medication_count", _count_medications(F.col(text_col)))
        df = df.withColumn("diagnosis_count", _count_diagnoses(F.col(text_col)))
        df = df.withColumn("procedure_count", _count_procedures(F.col(text_col)))
        df = df.withColumn("anatomy_count", _count_anatomy(F.col(text_col)))
        return df

    # ------------------------------------------------------------------ #
    #  Negation Detection
    # ------------------------------------------------------------------ #
    def extract_negation_features(
        self,
        notes_df: DataFrame,
        text_col: str = "clinical_notes",
    ) -> DataFrame:
        """Detect negated clinical terms.

        Adds columns: negation_count, negated_terms.
        """
        negation_cues = self._NEGATION_CUES

        @F.udf(IntegerType())
        def _negation_count(text):
            if text is None:
                return 0
            lower = text.lower()
            return sum(lower.count(cue) for cue in negation_cues)

        @F.udf(ArrayType(StringType()))
        def _negated_terms(text):
            if text is None:
                return []
            lower = text.lower()
            terms = []
            for cue in negation_cues:
                idx = 0
                while True:
                    pos = lower.find(cue, idx)
                    if pos == -1:
                        break
                    start = pos + len(cue)
                    # grab up to 4 words after the negation cue
                    snippet = lower[start:start + 60].split()[:4]
                    if snippet:
                        terms.append(" ".join(snippet))
                    idx = pos + 1
            return terms

        df = notes_df
        df = df.withColumn("negation_count", _negation_count(F.col(text_col)))
        df = df.withColumn("negated_terms", _negated_terms(F.col(text_col)))
        return df

    # ------------------------------------------------------------------ #
    #  Note Severity Score
    # ------------------------------------------------------------------ #
    def compute_note_severity_score(
        self,
        notes_df: DataFrame,
        text_col: str = "clinical_notes",
    ) -> DataFrame:
        """Score note severity as (critical_count - benign_count).

        Adds columns: critical_keyword_count, benign_keyword_count,
                      note_severity_score.
        """
        critical = self._CRITICAL_KEYWORDS
        benign = self._BENIGN_KEYWORDS

        @F.udf(IntegerType())
        def _count_critical(text):
            if text is None:
                return 0
            lower = text.lower()
            return sum(lower.count(kw) for kw in critical)

        @F.udf(IntegerType())
        def _count_benign(text):
            if text is None:
                return 0
            lower = text.lower()
            return sum(lower.count(kw) for kw in benign)

        df = notes_df
        df = df.withColumn("critical_keyword_count", _count_critical(F.col(text_col)))
        df = df.withColumn("benign_keyword_count", _count_benign(F.col(text_col)))
        df = df.withColumn(
            "note_severity_score",
            (F.col("critical_keyword_count") - F.col("benign_keyword_count")).cast(IntegerType()),
        )
        return df

    # ------------------------------------------------------------------ #
    #  BioClinicalBERT Embeddings
    # ------------------------------------------------------------------ #
    def extract_bert_embeddings(
        self,
        notes_df: DataFrame,
        text_col: str = "clinical_notes",
        id_col: str = "patient_id",
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> Optional[DataFrame]:
        """Generate 768-dim BioClinicalBERT embeddings for clinical notes.

        Collects text to the driver, runs inference in batches via
        ``transformers``, and returns a Spark DataFrame with columns
        [id_col, 'bert_embedding'].

        Falls back to None (caller should use TF-IDF) when torch /
        transformers are unavailable.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            print(
                "[NLPFeatureEngineer] torch/transformers not available. "
                "Skipping BERT embeddings; use TF-IDF instead."
            )
            return None

        # Collect to driver
        rows = notes_df.select(id_col, text_col).collect()
        ids = [row[id_col] for row in rows]
        texts = [row[text_col] or "" for row in rows]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)

            # Mean-pool over token dimension -> (batch, 768)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.extend(pooled.cpu().tolist())

        # Build Spark DataFrame
        spark = notes_df.sparkSession
        schema = StructType(
            [
                StructField(id_col, StringType(), False),
                StructField("bert_embedding", ArrayType(FloatType()), False),
            ]
        )
        embed_rows = [
            (str(pid), [float(v) for v in emb])
            for pid, emb in zip(ids, all_embeddings)
        ]
        return spark.createDataFrame(embed_rows, schema)
