"""Prompt-lookup draft: propose the tokens that followed the last earlier occurrence of the current ending.

No model and no cache. The last n tokens of seq (n from max_ngram down to min_ngram) are searched for in the earlier text, most
recent occurrence first, and the tokens after that occurrence become the proposal, up to k of them. When nothing matches, it
proposes nothing and the round is a plain one-token step. Structured outputs copy from the prompt (sentence ids, label words,
JSON keys), so matches are common there, but the acceptance rate is low because the guesses are crude. What it saves is cost:
a draft step takes about no time, where a 0.5B model draft takes 0.8 times a target step on the T4.
"""
import torch


class PromptLookupDraft:
    def __init__(self, max_ngram=3, min_ngram=1):
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.max_ngram, self.min_ngram, self.forwards = max_ngram, min_ngram, 0

    def start(self, prompt_ids):
        pass

    def truncate(self, valid):
        pass

    def propose(self, seq, k):
        """Up to k tokens (1-D long tensor, possibly empty) that followed the latest earlier match of the end of seq."""
        tokens = seq.tolist()
        for n in range(min(self.max_ngram, len(tokens) - 1), self.min_ngram - 1, -1):
            tail = tokens[-n:]
            for start in range(len(tokens) - n - 1, -1, -1):
                if tokens[start:start + n] == tail:
                    follow = tokens[start + n:start + n + k]
                    if follow:
                        return torch.tensor(follow, dtype=seq.dtype, device=seq.device)
        return seq[:0]
