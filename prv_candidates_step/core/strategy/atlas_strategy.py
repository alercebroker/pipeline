from .base_strategy import BasePrvCandidatesStrategy


class ATLASPrvCandidatesStrategy(BasePrvCandidatesStrategy):
    def process_prv_candidates(self, _: dict):
        return [], []
