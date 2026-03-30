import unittest
from collections import defaultdict

from bwt import TandemRepeatFinder


class RepeatAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finder = TandemRepeatFinder(
            "test2.fa",
            show_progress=False,
            max_motif_length=12,
        )
        sequences = cls.finder.load_reference()
        cls.finder.build_indices(sequences)
        cls.repeats = cls.finder.find_tandem_repeats(
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=False,
        )

        cls.by_chrom = defaultdict(list)
        for repeat in cls.repeats:
            cls.by_chrom[repeat.chrom].append(repeat)

    def _get_single_repeat(self, chrom_name: str):
        repeats = self.by_chrom[chrom_name]
        self.assertEqual(
            len(repeats),
            1,
            f"Expected a single repeat for {chrom_name}, found {len(repeats)}",
        )
        return repeats[0]

    def test_perfect_repeat_coordinates_and_motif(self):
        repeat = self._get_single_repeat("test1_PERFECT_7mer_5copies")
        self.assertEqual(repeat.start, 30)
        self.assertEqual(repeat.end, 65)
        self.assertEqual(repeat.motif, "TCATCGG")
        self.assertIsNone(repeat.variations)
        self.assertAlmostEqual(repeat.copies, 5.0)

    def test_interrupt_variations_are_reported(self):
        repeat = self._get_single_repeat("test4_INTERRUPTED_7mer_11copies")
        expected = {"6:5:C>A", "10:6:G>A", "11:0:ins(G)"}
        self.assertIsNotNone(repeat.variations)
        self.assertEqual(set(repeat.variations), expected)

    def test_duplicate_rotation_is_collapsed(self):
        motifs = {repeat.motif for repeat in self.by_chrom["test6_NESTED_long20_short4"]}
        self.assertIn("TGCTGATCGTAGCTAGCTGA", motifs)
        self.assertIn("TGCT", motifs)
        self.assertNotIn("CTGA", motifs)

    def test_long_repeat_reports_deletion_variations(self):
        repeat = self._get_single_repeat("test12_LONG_IMPERFECT_indel")
        self.assertIsNotNone(repeat.variations)
        self.assertTrue(
            any(op.startswith("9:10:del(") for op in repeat.variations),
            msg=f"Expected deletion variation in {repeat.variations}",
        )


if __name__ == "__main__":
    unittest.main()
