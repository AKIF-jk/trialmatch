# Eligibility screening report

> Educational screening prototype, not a clinical enrollment decision.

**Overall (rule-based): Likely ineligible**

- Criterion 4 (inclusion) is not included.

_Rule-based aggregation of the criterion labels, not a ground-truth verdict. Any not included or excluded gives Likely ineligible. Otherwise any not enough information (or unreadable output) gives Needs review. Otherwise Preliminarily eligible. not applicable criteria do not count._

## Criteria

### 1. Inclusion: Adults aged 18 to 75 years

- Label: **included** (output status: ok)
- Cited sentence ids: 0
  - [0] 58-year-old woman seen for a follow-up visit.
- Explanation (template from cited sentences): Labelled included for this inclusion criterion. Cited: [0] "58-year-old woman seen for a follow-up visit."
- Model rationale (generated from the cited sentences only, may be wrong): The label "included" follows from the sentence "58-year-old woman seen for a follow-up visit" because the inclusion criterion specifies that the study is for adults aged 18 to 75 years, and the 58-year-old woman falls within this age range.

### 2. Inclusion: Diagnosis of type 2 diabetes mellitus

- Label: **included** (output status: ok)
- Cited sentence ids: 1
  - [1] She has had type 2 diabetes for 9 years and takes metformin 1000 mg twice daily.
- Explanation (template from cited sentences): Labelled included for this inclusion criterion. Cited: [1] "She has had type 2 diabetes for 9 years and takes metformin 1000 mg twice daily."
- Model rationale (generated from the cited sentences only, may be wrong): The label "included" follows because the sentence states that the subject has had type 2 diabetes for 9 years and takes metformin 1000 mg twice daily, which indicates that they meet the inclusion criterion of having type 2 diabetes.

### 3. Inclusion: HbA1c between 7.5% and 11%

- Label: **not enough information** (output status: ok)
- Cited sentence ids: 2
  - [2] Her last HbA1c was 8.1%.
- Explanation (template from cited sentences): Labelled not enough information for this inclusion criterion. Cited: [2] "Her last HbA1c was 8.1%."
- Missing information: The note does not say enough to decide: HbA1c between 7.5% and 11%
- Model rationale (generated from the cited sentences only, may be wrong): The label "not enough information" indicates that the provided sentence does not contain the specific criterion for inclusion in the clinical trial, which is the HbA1c level between 7.5% and 11%.

### 4. Inclusion: History of cardiovascular disease

- Label: **not included** (output status: ok)
- Cited sentence ids: 5
  - [5] She is not pregnant and had a hysterectomy in 2015.
- Explanation (template from cited sentences): Labelled not included for this inclusion criterion. Cited: [5] "She is not pregnant and had a hysterectomy in 2015."
- Model rationale (generated from the cited sentences only, may be wrong): The label "not included" is based on the sentence "She is not pregnant and had a hysterectomy in 2015." Since the patient is not pregnant and has a history of a hysterectomy, they do not meet the inclusion criterion for cardiovascular disease, thus the label follows.

### 5. Exclusion: Pregnancy or breastfeeding

- Label: **not excluded** (output status: ok)
- Cited sentence ids: 5
  - [5] She is not pregnant and had a hysterectomy in 2015.
- Explanation (template from cited sentences): Labelled not excluded for this exclusion criterion. Cited: [5] "She is not pregnant and had a hysterectomy in 2015."
- Model rationale (generated from the cited sentences only, may be wrong): The label "not excluded" is based on the information that the subject is not pregnant and had a hysterectomy in 2015, which rules out pregnancy or breastfeeding as potential risks or contraindications for the clinical trial.

### 6. Exclusion: Serum creatinine above 1.5 mg/dL

- Label: **not enough information** (output status: ok)
- Cited sentence ids: 6
  - [6] Creatinine is 0.9 mg/dL.
- Explanation (template from cited sentences): Labelled not enough information for this exclusion criterion. Cited: [6] "Creatinine is 0.9 mg/dL."
- Missing information: The note does not say enough to decide: Serum creatinine above 1.5 mg/dL
- Model rationale (generated from the cited sentences only, may be wrong): The label "not enough information" indicates that the provided sentence does not contain the necessary criterion for exclusion, specifically the serum creatinine level above 1.5 mg/dL.

### 7. Exclusion: Current insulin therapy

- Label: **not excluded** (output status: ok)
- Cited sentence ids: none
- Explanation (template from cited sentences): Labelled not excluded for this exclusion criterion. No sentence was cited.
- Missing information: The label has no cited sentence, so nothing in the note was pointed to as support. Check it by hand.

## Patient note

0. 58-year-old woman seen for a follow-up visit.
1. She has had type 2 diabetes for 9 years and takes metformin 1000 mg twice daily.
2. Her last HbA1c was 8.1%.
3. Blood pressure today was 138/84.
4. She does not smoke.
5. She is not pregnant and had a hysterectomy in 2015.
6. Creatinine is 0.9 mg/dL.
7. She reports no chest pain or shortness of breath.

## Provenance

```json
{
  "app_commit": "56d88c392b50a677d134251cff8611e1e2256aa9",
  "timestamp_utc": "2026-11-02T11:47:54+00:00",
  "target_model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
  },
  "target_adapter": {
    "repo": "AkifJK/trialmatch-qwen2.5-1.5b-lora",
    "revision": "6fd2471b3894cdbe84d46f9d1278233a3279e77e",
    "folder": "seed44",
    "sha256": "6d6cc082e60431ee6accf28146b58d03f51e966ec16a1866d19364a79f535034",
    "note": "over4_s44 (highest validation macro-F1, 0.473)"
  },
  "draft_model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "7ae557604adf67be50417f59c2c2f167def9a775"
  },
  "draft_adapter": {
    "folder": "draft",
    "sha256": "91886b965f3a03a3cb5e86ed080c76025a086fbc865e56fa264db37a34388d44"
  },
  "decoding": {
    "do_sample": false,
    "repetition_penalty": 1.0,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "max_new_tokens": 64,
    "decoder": "decode_cached",
    "attention": "sdpa"
  },
  "device": "Tesla T4",
  "dtype": "float16",
  "packages": {
    "torch": "2.11.0+cu128",
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "datasets": "4.8.5",
    "gradio": "5.50.0",
    "numpy": "2.1.3",
    "scikit-learn": "1.6.1"
  }
}
```

> Educational screening prototype, not a clinical enrollment decision.
