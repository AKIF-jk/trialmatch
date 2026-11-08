# Error analysis on the test split

Written after the one-shot test run (tag `final-freeze`, commit `457731a`). It reads the saved outputs in `results/final/test/` and changes nothing about the models, prompts or scoring. Regenerate the counts with `PYTHONPATH=. python scripts/error_analysis.py` (writes `results/final/analysis/`). The expert annotation is the reference throughout, even where it is debatable.

## What the counts say

The test split has 206 rows from 8 patients. Label errors per seed: 60 (seed 42), 64 (seed 43), 64 (seed 44). **55 rows are wrong for all three seeds**, so most mistakes are systematic, not seed noise.

For seed 44, the deployed adapter, the 64 label errors are:

| Gold label -> predicted | Rows |
|---|---|
| not enough information -> not excluded | 27 |
| not applicable -> not excluded | 9 |
| included -> not enough information | 7 |
| not enough information -> not included | 7 |
| not included -> not enough information | 5 |
| included -> not included | 3 |
| not applicable -> not enough information | 3 |
| not excluded -> excluded, not applicable, not enough information (one each) | 3 |

- **The model almost never says "not enough information" for an exclusion criterion.** Of 28 exclusion rows the experts marked "not enough information", seed 44 gets 1 right (seeds 42 and 43 get 1 and 2). It predicts that label on only 2 exclusion rows in total. In training the label is not rare for exclusions (65 of 387 rows, 17%). So 36 of 64 errors (27 plus 9) are exclusion criteria answered "not excluded" when the experts said "not enough information" or "not applicable". Why the model leans this way was not tested; a closed-world habit ("nothing mentioned, so not excluded") is a guess.
- **Rare classes are mostly missed.** The test split has no `excluded` row, so that class cannot be scored. Of 7 `not included` rows, seed 44 gets 2 right (F1 0.17 to 0.29 across the three seeds). Of 15 `not applicable` rows, 3 are right for seed 44 (F1 0.00 to 0.32 across seeds).
- **Evidence.** Among rows whose label is right, seed 44 cites at least one sentence the expert did not on 13 of 142. It cites nothing where the expert cited something on 19 rows, and cites something where the expert cited nothing on 14. When the model says "not enough information" it often still cites sentences (19 rows). The experts do the same: 17 of the 66 gold "not enough information" rows carry evidence, so this is not always a mistake.
- **Decisions without evidence.** Seed 44 says included, not included or excluded without citing any sentence on 3 rows (ids 161, 176, 746).

## Ten errors, by kind

Annotation ids are the dataset's `annotation_id`. "All seeds" means the label is wrong for seeds 42, 43 and 44. The explanations are readings of the text, not tested causes.

**Missing information treated as absence**

1. **id 174**, trial NCT02097186, exclusion "Patients with previous axillary surgery". Gold: not enough information, no evidence. Predicted: not excluded, no evidence. All seeds. Neither the experts nor the model cite a sentence, and the model resolves the gap as "no".
2. **id 193**, NCT01526382, exclusion "Patients were moribund". Gold: not enough information, citing sentences 0 to 4 (acute arm pain, pallor, tachycardia, blood pressure 80/60, an arm without discoloration). Predicted: not excluded, no evidence. All seeds. The experts cite findings that bear on the criterion but do not settle it; the model cites nothing.

**Rare classes**

3. **ids 683 and 693**, patient `sigir-20158`, trials NCT02562040 and NCT00393913 (exclusions for previous tonsillectomy and for other sleep disorders). Gold: not applicable. Predicted: not excluded, no evidence. All seeds. The model does not separate "does not apply to this patient" from "applies and is not met".
4. **id 322**, NCT01745731, exclusion "Pregnant women or women of childbearing age who do not have adequate contraception", for a 48-year-old man. Gold: not excluded, citing sentence 0. Seed 44 predicts not applicable, no evidence. Not wrong for all seeds. The boundary between the two labels for sex-specific criteria is a judgement call in the annotation.
5. **id 162**, NCT02097186, inclusion "Patients undergoing open abdominal aortic aneurysm repair". Gold: not included, no evidence. Predicted: not enough information, citing sentences 0 to 2 (a groin mass and bruit after catheterization). All seeds. The cited text describes a different vascular problem, so the evidence does not support the criterion. Ids 163 and 164, from the same trial, show the same pattern. Five of the seven gold `not included` rows end up as "not enough information" for seed 44.

**Contradictions between the label and the cited evidence**

6. **id 186**, NCT01526382, inclusion "Heart rate of at least 90/min". Gold: included, citing sentence 2. Predicted: not enough information, citing sentence 2 ("tachypneic and tachycardic"). All seeds. The model cites the sentence the experts cite and then does not accept it as sufficient.
7. **id 383**, NCT00997100, inclusion "active SLE disease with at least one of the following symptoms" (a list that includes arthritis and rash). Gold: included, citing sentences 0 and 1. Predicted: not included, citing sentences 0, 1 and 2. All seeds. The cited sentences describe joint pain, a facial rash and swollen wrists, which are items on the criterion's own list, yet the label says the opposite.
8. **id 176**, NCT02264964, inclusion "No medical history of central vena catheterization". Gold: included, citing sentence 0 (a cardiac catheterization through the femoral artery). Predicted: not included, no evidence. All seeds. A negated criterion plus a decision with no cited sentence.

**Unsupported or excess evidence**

9. **id 746**, NCT01255670, inclusion "patient has peritonsillar abscess". Gold: not enough information, citing sentence 0 (a child with dysphagia, drooling and fever). Predicted: not included, no evidence. Not wrong for all seeds. A firm negative with nothing cited, where the experts saw suggestive but inconclusive findings.
10. **id 184**, NCT02264964, exclusion "Other inappropriate situation". Gold: not excluded, no evidence. Predicted: excluded, citing sentences 0 to 2 (the catheterization complication). This is the only `excluded` prediction of seed 44 on the test split, and since no test row is `excluded`, that label has zero precision. A catch-all criterion invites reading any complication as an exclusion.

Smaller cases of the same family: ids 306 and 384 have the right label but cite one sentence more than the experts ("taken to the ICU"; a laboratory sentence for a criterion about specific laboratory thresholds).

## Limits

- 8 patients and 206 rows, so a single patient can move several counts together. Several cases above come from the same trial, and ids 162 to 164, 174, 176 and 184 are all one patient's note.
- The ten cases were chosen to cover each kind of error, not drawn at random. The counts in the first section cover all 206 rows.
- Some expert labels are debatable (ids 322 and 413 for example), and there is a single annotator per row in the data we use, so "wrong" means "differs from that annotation".
- No ablation was run to test any of the explanations. Changing prompts or training after seeing these results would break the freeze, so none of this is acted on in the reported system.
