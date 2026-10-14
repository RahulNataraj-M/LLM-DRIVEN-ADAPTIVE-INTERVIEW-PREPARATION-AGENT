import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402


class ProfileTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="ip_profile_test_")
        cls._saved = {}
        for attr, sub in (("CANDIDATE_DIR", os.path.join(cls._tmp, "candidates")),
                          ("VECTORS_DIR", os.path.join(cls._tmp, "vectors")),
                          ("RL_TABLES_DIR", os.path.join(cls._tmp, "rl", "tables")),
                          ("SESSION_DIR", os.path.join(cls._tmp, "sessions")),
                          ("LOGS_DIR", os.path.join(cls._tmp, "logs"))):
            cls._saved[attr] = getattr(C, attr)
            setattr(C, attr, sub)

    @classmethod
    def tearDownClass(cls):
        for attr, val in cls._saved.items():
            setattr(C, attr, val)

    def tearDown(self):
        for name in os.listdir(self._tmp) if os.path.isdir(self._tmp) else []:
            p = os.path.join(self._tmp, name)
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p, ignore_errors=True)


class TestNewProfile(ProfileTestBase):
    def test_defaults(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("alice")
        d = p.to_dict()
        for field in ("skills", "subtopic_skills", "attempts", "hints_used",
                      "recent_outcomes", "mistakes", "misconceptions", "strengths",
                      "weaknesses", "topic_history", "difficulty_history", "behavior",
                      "progress", "performance", "last_seen"):
            self.assertIn(field, d)
        self.assertTrue(all(v == 0.5 for v in d["skills"].values()))
        self.assertTrue(all(s == 0.5
                            for subs in d["subtopic_skills"].values() for s in subs.values()))
        self.assertEqual(p.misconceptions, {})
        self.assertEqual(p.strengths, {})
        self.assertEqual(p.weaknesses, {})
        self.assertEqual(p.progress["score_history"], [])


class TestFirstAnswer(ProfileTestBase):
    def test_correct(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c1")
        p.record("DSA", 0.9, hinted=False, question="Q1")
        self.assertGreater(p.skills["DSA"], 0.5)          # skill moved up from 0.5
        self.assertEqual(p.attempts["DSA"], 1)            # attempt incremented
        self.assertEqual(p.recent_outcomes, [0.9])        # outcome recorded
        self.assertEqual(p.topic_history["DSA"]["attempts"], 1)
        self.assertEqual(p.topic_history["DSA"]["correct"], 1)


class TestIncorrectAnswer(ProfileTestBase):
    def test_wrong(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c2")
        p.record("DSA", 0.2, hinted=True, question="Hard graph question")
        self.assertLess(p.skills["DSA"], 0.5)             # skill decreased
        self.assertEqual(len(p.mistakes), 1)              # mistake recorded
        m = p.mistakes[0]
        self.assertEqual(m["topic"], "DSA")
        self.assertFalse(m["resolved"])
        self.assertEqual(p.hints_used["DSA"], 1)
        self.assertNotIn("DSA", p.weaknesses)             # not weak after one failure
        # weakness appears after repeated failure
        p.record("DSA", 0.2, question="Another hard one")
        self.assertIn("DSA", p.weaknesses)
        self.assertEqual(p.weaknesses["DSA"]["evidence_count"], 2)
        self.assertIn(p.weaknesses["DSA"]["severity"], ("low", "medium", "high"))


class TestRepeatedPerformance(ProfileTestBase):
    def test_no_classification_after_single(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c3")
        p.record("OOP", 0.95, question="Good OOP answer")
        self.assertEqual(p.strengths, {})

    def test_strength_after_repeated(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c4")
        for _ in range(8):
            p.record("OOP", 0.9, question="OOP question")
        self.assertIn("OOP", p.strengths)
        self.assertGreaterEqual(p.strengths["OOP"]["evidence_count"], 3)
        self.assertGreaterEqual(p.strengths["OOP"]["score"], 0.75)


class TestSubtopic(ProfileTestBase):
    def test_subtopic_updates(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c5")
        before_sub = p.subtopic_skills["DSA"]["Graphs"]
        before_topic = p.skills["DSA"]
        p.record("DSA", 0.9, subtopic="Graphs", question="Graph traversal")
        self.assertGreater(p.skills["DSA"], before_topic)                      # topic updated
        self.assertGreater(p.subtopic_skills["DSA"]["Graphs"], before_sub)     # subtopic updated
        self.assertEqual(p.subtopic_attempts["DSA"]["Graphs"], 1)

    def test_unknown_subtopic_safe(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c6")
        p.record("DSA", 0.9, subtopic="Quantum Computing", question="Q")
        self.assertEqual(p.subtopic_attempts["DSA"]["Graphs"], 0)  # nothing corrupted
        self.assertGreater(p.skills["DSA"], 0.5)                   # topic still tracked

    def test_resolve_subtopic(self):
        from candidate.profile import resolve_subtopic
        self.assertEqual(resolve_subtopic("DSA", "detect a cycle in a singly linked list"),
                         "Linked Lists")
        self.assertEqual(resolve_subtopic("DSA", "anything", explicit="Graphs"), "Graphs")
        self.assertIsNone(resolve_subtopic("DSA", "absolutely nothing relevant here"))


class TestDifficultyHistory(ProfileTestBase):
    def test_multiple_levels(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c7")
        p.record("DSA", 0.9, difficulty=1, question="Q1")
        p.record("DSA", 0.9, difficulty=2, question="Q2.1")
        p.record("DSA", 0.2, difficulty=2, question="Q2.2")
        p.record("DSA", 0.4, difficulty=3, question="Q3")
        self.assertEqual(p.difficulty_history["1"], {"attempts": 1, "correct": 1, "accuracy": 1.0})
        self.assertEqual(p.difficulty_history["2"]["attempts"], 2)
        self.assertEqual(p.difficulty_history["2"]["correct"], 1)
        self.assertEqual(p.difficulty_history["3"]["accuracy"], 0.0)
        self.assertEqual(set(p.difficulty_history), {"1", "2", "3"})


class TestMisconception(ProfileTestBase):
    def test_stored_from_evaluator_only(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c8")
        # no evaluator signal -> nothing invented
        p.record("DSA", 0.3, question="binary search", misconceptions=[])
        self.assertEqual(p.misconceptions, {})
        # an explicit evaluator misconception is stored
        p.record("DSA", 0.3, subtopic="Arrays", question="binary search",
                 misconceptions=["Candidate believes binary search applies to any unsorted array"])
        bucket = p.misconceptions["DSA"]
        self.assertEqual(len(bucket), 1)
        self.assertEqual(bucket[0]["occurrences"], 1)
        self.assertEqual(bucket[0]["concept"], "Arrays")
        self.assertFalse(bucket[0]["resolved"])
        # repeat increases occurrences + confidence
        p.record("DSA", 0.3, subtopic="Arrays", question="binary search again",
                 misconceptions=["Candidate believes binary search applies to any unsorted array"])
        self.assertEqual(p.misconceptions["DSA"][0]["occurrences"], 2)
        self.assertGreater(p.misconceptions["DSA"][0]["confidence"], 0.70)


class TestPersistence(ProfileTestBase):
    def test_roundtrip(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c9", name="Rahul")
        p.record("DSA", 0.9, subtopic="Graphs", difficulty=3, question="Q1",
                 misconceptions=["mixup1"])
        p.record_session(0.7)
        p.save()

        q = CandidateProfile.load("c9")
        self.assertEqual(q.candidate_id, "c9")
        self.assertEqual(q.skills, p.skills)
        self.assertEqual(q.subtopic_skills, p.subtopic_skills)
        self.assertEqual(q.attempts, p.attempts)
        self.assertEqual(q.recent_outcomes, p.recent_outcomes)
        self.assertEqual(q.mistakes, p.mistakes)
        self.assertEqual(q.misconceptions, p.misconceptions)
        self.assertEqual(q.topic_history, p.topic_history)
        self.assertEqual(q.difficulty_history, p.difficulty_history)
        self.assertEqual(q.progress, p.progress)
        self.assertIn("hints_per_question", q.behavior)
        self.assertEqual(q.strengths, p.strengths)
        self.assertEqual(q.weaknesses, p.weaknesses)


class TestMigration(ProfileTestBase):
    def test_legacy_load(self):
        from candidate.profile import CandidateProfile
        os.makedirs(C.CANDIDATE_DIR, exist_ok=True)
        legacy = {
            "candidate_id": "legacy_user",
            "skills": {"DSA": 0.48},
            "attempts": {"DSA": 3},
            "hints_used": {"DSA": 1},
            "recent_outcomes": [0.5, 0.6, 0.4],
            "mistakes": [{"topic": "DSA", "question": "old", "gist": "g", "ts": "2026-01-01"}],
            "last_seen": "2026-01-01T00:00:00",
        }
        with open(os.path.join(C.CANDIDATE_DIR, "legacy_user.json"), "w") as f:
            json.dump(legacy, f)
        p = CandidateProfile.load("legacy_user")
        self.assertEqual(p.skills["DSA"], 0.48)                       # old data kept
        self.assertEqual(p.skills["DBMS"], 0.5)                       # migrated defaults
        self.assertTrue(all(s == 0.5 for subs in p.subtopic_skills.values() for s in subs.values()))
        self.assertEqual(p.misconceptions, {})
        self.assertEqual(p.strengths, {})
        self.assertEqual(p.topic_history, {})
        self.assertEqual(p.difficulty_history, {})
        self.assertEqual(p.progress["score_history"], [])
        self.assertEqual(len(p.mistakes), 1)
        self.assertFalse(p.mistakes[0]["resolved"])                   # normalized
        self.assertIsNone(p.mistakes[0]["subtopic"])


class TestRLCompatibility(ProfileTestBase):
    def test_state(self):
        from candidate.profile import CandidateProfile
        from rl.q_learning import QLearner
        from config import RL_ACTIONS
        p = CandidateProfile.create("c10")
        p.record("DSA", 0.2, question="x")
        band = p.skill_band()
        self.assertIn(band, (0, 1, 2))
        st = p.rl_state("DSA", 3)
        self.assertEqual(st["skill_band"], band)
        self.assertIn(st["difficulty"], (1, 2, 3, 4, 5))
        self.assertIn("topic_idx", st)
        learner = QLearner.load(table_id="c10_or_fresh")
        action = learner.choose(band, 3, 0)
        self.assertIn(action, RL_ACTIONS)
        # Q update still works with the profile-derived band
        learner.update(band, 3, 0, action, 0.5, 3)
        learner.save()


class TestMLCompatibility(ProfileTestBase):
    def test_features_predictor(self):
        from candidate.profile import CandidateProfile
        from ml.features import build_features, FEATURES
        from ml.performance import PerformancePredictor
        p = CandidateProfile.create("c11")
        p.record("DSA", 0.8, question="q", hinted=False)
        feat = build_features(p, "DSA", 2, False)
        self.assertEqual(len(feat), len(FEATURES))
        self.assertTrue(all(isinstance(v, float) for v in feat))
        pred = PerformancePredictor.load()
        p_ok, used_ml = pred.predict_accuracy(feat)
        self.assertGreaterEqual(p_ok, 0.0)
        self.assertLessEqual(p_ok, 1.0)
        self.assertIsInstance(used_ml, bool)


class TestSessionProgressAndViews(ProfileTestBase):
    def test_progress_and_views(self):
        from candidate.profile import CandidateProfile
        p = CandidateProfile.create("c12")
        p.record("DSA", 0.8, question="q1")
        p.record_session(0.7)
        p.record_session(0.8)
        self.assertEqual(p.progress["sessions_completed"], 2)
        self.assertEqual(p.progress["score_history"], [0.7, 0.8])
        self.assertAlmostEqual(p.progress["improvement"], 0.1)
        perf = p.performance()
        self.assertEqual(perf["total_questions"], 1)
        self.assertIn("average_difficulty", perf)
        self.assertTrue(p.prompt_context("DSA"))
        self.assertIn("DSA", p.to_dict()["skills"])
        self.assertIn("recent_accuracy", p.summary())
        self.assertIn("candidate_id", p.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
