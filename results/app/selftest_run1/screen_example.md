# Eligibility screening report

> Educational screening prototype, not a clinical enrollment decision.

**Overall (rule-based): Preliminarily eligible**

- No criterion is blocking and none needs more information.

_Rule-based aggregation of the criterion labels, not a ground-truth verdict. Any not included or excluded gives Likely ineligible. Otherwise any not enough information (or unreadable output) gives Needs review. Otherwise Preliminarily eligible. not applicable criteria do not count._

## Criteria

### 1. Inclusion: Adults aged 18 to 75 years

- Label: **included** (output status: ok)
- Cited sentence ids: 0
  - [0] 58-year-old woman seen for a follow-up visit.
- Explanation (template from cited sentences): Labelled included for this inclusion criterion. Cited: [0] "58-year-old woman seen for a follow-up visit."

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
  "timestamp_utc": "2026-11-02T11:41:10+00:00",
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
