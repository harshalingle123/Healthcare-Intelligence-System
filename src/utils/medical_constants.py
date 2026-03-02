"""
Medical Constants for Healthcare Intelligence System
=====================================================
Dictionaries and lists of medical abbreviations, symptom-to-system mappings,
diagnosis categories, clinical keywords, and common drug names.
"""

# --------------------------------------------------------------------------- #
#  Medical Abbreviations (>= 30 entries)
# --------------------------------------------------------------------------- #
MEDICAL_ABBREVIATIONS = {
    "pt": "patient",
    "hx": "history",
    "dx": "diagnosis",
    "rx": "prescription",
    "tx": "treatment",
    "sx": "symptoms",
    "fx": "fracture",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "o2": "oxygen saturation",
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "hgb": "hemoglobin",
    "plt": "platelets",
    "bun": "blood urea nitrogen",
    "cr": "creatinine",
    "na": "sodium",
    "k": "potassium",
    "ca": "calcium",
    "mg": "magnesium",
    "ekg": "electrocardiogram",
    "cxr": "chest x-ray",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "prn": "as needed",
    "bid": "twice a day",
    "tid": "three times a day",
    "qid": "four times a day",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "npo": "nothing by mouth",
    "abd": "abdomen",
    "ams": "altered mental status",
    "bmp": "basic metabolic panel",
    "cbc": "complete blood count",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "ckd": "chronic kidney disease",
    "gerd": "gastroesophageal reflux disease",
}

# --------------------------------------------------------------------------- #
#  Symptom-to-Organ-System Mapping
# --------------------------------------------------------------------------- #
SYMPTOM_TO_SYSTEM = {
    # Respiratory
    "cough": "respiratory",
    "shortness_of_breath": "respiratory",
    "wheezing": "respiratory",
    "sputum_production": "respiratory",
    "hemoptysis": "respiratory",
    # Cardiovascular
    "chest_pain": "cardiovascular",
    "palpitations": "cardiovascular",
    "edema": "cardiovascular",
    "syncope": "cardiovascular",
    "orthopnea": "cardiovascular",
    # Gastrointestinal
    "nausea": "gastrointestinal",
    "vomiting": "gastrointestinal",
    "abdominal_pain": "gastrointestinal",
    "diarrhea": "gastrointestinal",
    "constipation": "gastrointestinal",
    # Neurological
    "headache": "neurological",
    "dizziness": "neurological",
    "seizure": "neurological",
    "confusion": "neurological",
    "numbness": "neurological",
    # Musculoskeletal
    "joint_pain": "musculoskeletal",
    "muscle_weakness": "musculoskeletal",
    "back_pain": "musculoskeletal",
    "swelling": "musculoskeletal",
    # General / Systemic
    "fever": "systemic",
    "fatigue": "systemic",
    "weight_loss": "systemic",
    "night_sweats": "systemic",
    "malaise": "systemic",
    # Metabolic
    "excessive_thirst": "metabolic",
    "frequent_urination": "metabolic",
    "blurred_vision": "metabolic",
}

# --------------------------------------------------------------------------- #
#  Diagnosis Categories with ICD-10 Codes
# --------------------------------------------------------------------------- #
DIAGNOSIS_CATEGORIES = [
    {
        "category": "Respiratory Disease",
        "icd10_prefix": "J00-J99",
        "examples": ["J44.1 - COPD with acute exacerbation",
                      "J18.9 - Pneumonia, unspecified"],
        "label_index": 0,
    },
    {
        "category": "Cardiovascular Disease",
        "icd10_prefix": "I00-I99",
        "examples": ["I21.9 - Acute myocardial infarction",
                      "I50.9 - Heart failure, unspecified"],
        "label_index": 1,
    },
    {
        "category": "Metabolic / Endocrine Disorder",
        "icd10_prefix": "E00-E89",
        "examples": ["E11.9 - Type 2 diabetes mellitus",
                      "E78.5 - Hyperlipidemia"],
        "label_index": 2,
    },
    {
        "category": "Gastrointestinal Disorder",
        "icd10_prefix": "K00-K95",
        "examples": ["K21.0 - GERD with esophagitis",
                      "K35.80 - Acute appendicitis"],
        "label_index": 3,
    },
    {
        "category": "Neurological Disorder",
        "icd10_prefix": "G00-G99",
        "examples": ["G43.909 - Migraine, unspecified",
                      "G40.909 - Epilepsy, unspecified"],
        "label_index": 4,
    },
]

# --------------------------------------------------------------------------- #
#  Clinical NLP Keywords
# --------------------------------------------------------------------------- #
CRITICAL_KEYWORDS = [
    "emergent", "critical", "unstable", "deteriorating", "acute",
    "severe", "life-threatening", "urgent", "hemorrhage", "shock",
    "sepsis", "respiratory failure", "cardiac arrest", "unresponsive",
    "intubation", "code blue", "stat", "decompensating", "anaphylaxis",
    "stroke", "embolism", "infarction", "ischemia", "coma",
    "renal failure", "multi-organ failure", "disseminated", "tension",
    "perforation", "rupture", "tamponade", "status epilepticus",
]

BENIGN_KEYWORDS = [
    "stable", "improved", "resolved", "mild", "benign",
    "normal", "unremarkable", "intact", "well-appearing", "afebrile",
    "tolerating", "ambulatory", "oriented", "comfortable", "healing",
    "routine", "prophylactic", "maintenance", "follow-up", "outpatient",
    "discharged", "chronic", "controlled", "uncomplicated", "no acute",
]

# --------------------------------------------------------------------------- #
#  Common Drug Names (>= 50 entries)
# --------------------------------------------------------------------------- #
DRUG_NAMES = [
    # Cardiovascular
    "lisinopril", "amlodipine", "losartan", "metoprolol", "atenolol",
    "carvedilol", "valsartan", "diltiazem", "hydralazine", "furosemide",
    "hydrochlorothiazide", "spironolactone", "warfarin", "clopidogrel",
    "aspirin", "heparin", "enoxaparin", "atorvastatin", "simvastatin",
    "rosuvastatin",
    # Diabetes / Metabolic
    "metformin", "glipizide", "insulin", "sitagliptin", "empagliflozin",
    "liraglutide", "pioglitazone", "levothyroxine",
    # Respiratory
    "albuterol", "fluticasone", "montelukast", "ipratropium",
    "budesonide", "prednisone", "dexamethasone",
    # Pain / Anti-inflammatory
    "ibuprofen", "acetaminophen", "naproxen", "morphine", "hydrocodone",
    "oxycodone", "tramadol", "gabapentin", "pregabalin", "celecoxib",
    # Antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "metronidazole", "ceftriaxone", "vancomycin", "piperacillin",
    "trimethoprim", "levofloxacin", "clindamycin",
    # GI
    "omeprazole", "pantoprazole", "ondansetron", "famotidine",
    # Psychiatry / Neuro
    "sertraline", "fluoxetine", "escitalopram", "duloxetine",
    "lorazepam", "diazepam", "alprazolam", "quetiapine",
    "levetiracetam", "phenytoin",
]
