# Hand-written unseen cases

Three patients and 26 criteria, written for this project and kept in `configs/unseen_cases.json`. None of the sentences or criteria comes from the dataset, and a test checks that. They never touched the training, validation or test split. Run from commit `33e87e2` on a Tesla T4 with the same decoding as the final evaluation (`scripts/unseen_eval.py`, output in `results/final/unseen/`).

**Authorship caveat.** The expected labels and evidence were written by the assistant (Claude) following the dataset's conventions as described in the prompt. A clinician did not review them, and some are debatable. Read the numbers as a sanity check on unfamiliar notes, not as a benchmark. With 26 criteria one criterion moves accuracy by 0.04.

## Result

| Model | Inclusion | Exclusion | All | Label accuracy | Evidence macro-F1 |
|---|---|---|---|---|---|
| seed 42 | 9/13 | 5/13 | 14/26 | 0.538 | 0.827 |
| seed 43 | 10/13 | 5/13 | 15/26 | 0.577 | 0.865 |
| seed 44 | 9/13 | 6/13 | 15/26 | 0.577 | 0.865 |

No output was malformed. Label accuracy is lower than on the test split (0.689 to 0.709), and the sample is small enough that the gap is within noise. What carries over is the pattern of mistakes.

## What repeats from the test split

- **Exclusion criteria with no information are answered "not excluded".** The three criteria whose expected label is "not enough information" (macrolide allergy, cluster headache, active infection) get "not excluded" from all three seeds. On the test split seed 44 gets 1 of 28 such rows right.
- **`excluded` is never predicted.** Both expected `excluded` criteria (hospital admission in the previous 30 days, hormonal contraception) are answered "not excluded" by all three seeds (no seed predicts `excluded` anywhere in the 26), even though the model cites the sentence that triggers the exclusion ("discharged ... two weeks ago", "using an oral contraceptive pill"). That is a label that contradicts its own cited evidence, the same family as ids 186 and 383 in `docs/error_analysis.md`.
- **Not applicable is mostly missed.** "Negative pregnancy test" for a man is answered "not enough information", and "Benign prostatic hyperplasia" for a woman is answered "not excluded", by every seed. "Current pregnancy" for a man is right for seed 44 only.
- **Inference from a negated or absent mention is shaky.** "Measurable metastatic disease" (the note says no distant metastases) gets "not enough information" from all seeds, and "Uncontrolled hypertension" with a blood pressure of 118/74 gets "not enough information" from all three seeds, while citing the blood-pressure sentence.

## What works

Criteria whose answer is stated in one sentence are mostly right: age, radiographic pneumonia, anticoagulant use, creatinine, ECOG status, ejection fraction, prior chemotherapy, and aura. The cited sentence is usually the expected one (evidence macro-F1 0.83 to 0.87).

## Limits

- 26 criteria on 3 patients, single labeller (the assistant), no clinician review.
- The three patients are in three different specialities, none of them close to the TrialGPT annotation set, so this mixes "unfamiliar notes" with "unfamiliar wording".
- Nothing here was used to change the model, prompt or scoring.
