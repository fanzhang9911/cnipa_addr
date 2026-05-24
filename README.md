# CNIPA Applicant Address

This repository provides the code used to construct a structured applicant-address dataset for Chinese patent applications from CNIPA data.

The workflow includes applicant classification, country/region code imputation, applicant-address imputation, geocoding with Amap and LLM-based methods, and final assembly of geocoded applicant-address records.

## Repository Structure

```text
cnipa_addr/
│
├── 1_applicant_classification.py
├── 1_countrycode_imputation.py
│
├── 2_imputation_by_agapi.py
├── 2_imputation_by_registration_addr.py
├── 2_imputation_within_cnipa.py
│
├── 3_geoc_by_amap.py
├── 3_geoc_by_llm.py
│
├── 4_assembly.py
│
└── gcj2wgs.py
