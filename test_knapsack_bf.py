import unittest
import knapsack_class
import numpy as np
import knapsack_vk


class TestBruteForce(unittest.TestCase):
    def test_all_equal(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                4, np.array([5, 8, 7, 3]), np.array([3, 6, 2, 4]), 10
            )
        )
        self.assertEqual(solution, (7, [1, 0, 0, 1]))

    def test_capacity_equal_zero(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                4, np.array([5, 8, 7, 3]), np.array([3, 6, 2, 4]), 0
            )
        )
        self.assertEqual(solution, (0, [0, 0, 0, 0]))

    def test_capacity_sup_sum_weight(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                6, np.array([5, 8, 7, 3, 12, 2]), np.array([3, 6, 2, 4, 4, 1]), 37
            )
        )
        self.assertEqual(solution, (20, [1, 1, 1, 1, 1, 1]))

    def test_only_nb_1(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                6, np.array([8, 8, 7, 3, 12, 2]), np.array([40, 6, 2, 4, 4, 1]), 8
            )
        )
        self.assertEqual(solution, (40, [1, 0, 0, 0, 0, 0]))

    def test_only_nb_2(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                6, np.array([8, 8, 7, 3, 12, 2]), np.array([6, 40, 2, 4, 4, 1]), 8
            )
        )
        self.assertEqual(solution, (40, [0, 1, 0, 0, 0, 0]))

    def test_no_obj(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(0, np.array([]), np.array([]), 8)
        )
        self.assertEqual(solution, (0, []))

    def test_anti_glouton_p(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                6, np.array([8, 8, 4, 4, 12, 2]), np.array([6, 40, 22, 22, 4, 1]), 8
            )
        )
        self.assertEqual(solution, (44, [0, 0, 1, 1, 0, 0]))

    def test_anti_glouton_ratio_p_w(self):
        solution = knapsack_vk.bruteforce(
            knapsack_class.Knapsack(
                4, np.array([7, 8, 4, 4]), np.array([15, 10, 8, 8]), 8
            )
        )
        self.assertEqual(solution, (16, [0, 0, 1, 1]))


if __name__ == "__main__":
    unittest.main()
