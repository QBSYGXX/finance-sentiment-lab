import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import normalize, MAX_LENGTH
from model import TextCNN, build_vocab
from prepare import clean_splits


class InputAndSplitTests(unittest.TestCase):
    def test_model_input_normalization(self):
        self.assertEqual(normalize('  Ａ股\n 看好  '), 'A股 看好')
        self.assertEqual(len(normalize('涨'*200)), MAX_LENGTH)

    def test_conflicts_and_overlaps_are_not_leaked(self):
        train = [[f'训练样本{i}类别{label}', label] for label in range(3) for i in range(20)]
        test = [[f'测试样本{i}类别{label}', label] for label in range(3) for i in range(3)]
        train += [['交叉重复', 2], ['标签冲突', 0], ['同组冲突', 0], ['同组冲突', 1]]
        test += [['交叉重复', 2], ['标签冲突', 1]]
        splits, audit, _ = clean_splits(train, test)
        sets = {k: {r['text'] for r in rows} for k, rows in splits.items()}
        self.assertIn('交叉重复', sets['test'])
        self.assertNotIn('交叉重复', sets['train'] | sets['validation'])
        self.assertNotIn('标签冲突', set.union(*sets.values()))
        self.assertEqual(audit['conflicting_texts_removed'], 2)
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])

    def test_truncated_duplicate_inputs_are_grouped(self):
        train = [[f'训练{i}类{j}', j] for i in range(20) for j in range(3)]
        test = [['A'*128 + '不同尾巴', 2]]
        train += [['A'*128 + '训练尾巴', 2]]
        splits, _, _ = clean_splits(train, test)
        self.assertFalse(any(r['text'] == 'A'*128 for r in splits['train'] + splits['validation']))

    def test_padding_cannot_change_prediction(self):
        torch.manual_seed(42)
        model = TextCNN(20).eval()
        with torch.no_grad():
            a = model(torch.tensor([[2,3,4,5,0,0]]))
            b = model(torch.tensor([[2,3,4,5,0,0,0,0,0,0]]))
        torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
