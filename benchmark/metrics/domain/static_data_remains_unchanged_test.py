import unittest
import pandas as pd

from benchmark import PLAYER_ID
from benchmark.utils import FIRST_NAME, LAST_NAME, PSEUDONYM, HEIGHT, DATE_OF_BIRTH, POSITION, CITIZENSHIP, FOOT
from .static_data_remains_unchanged import StaticDataRemainsUnchanged


class TestStaticDataRemainsUnchanged(unittest.TestCase):

    def setUp(self):
        self.metric = StaticDataRemainsUnchanged()

    def test_no_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            FIRST_NAME: ["John", "John", "Mike", "Mike"],
            LAST_NAME: ["Doe", "Doe", "Smith", "Smith"],
            PSEUDONYM: ["JD", "JD", "MS", "MS"],
            HEIGHT: [180, 180, 175, 175],
            DATE_OF_BIRTH: ["1990-01-01", "1990-01-01", "1985-05-20", "1985-05-20"],
            FOOT: ["Right", "Right", "Left", "Left"],
            POSITION: ["Midfielder", "Midfielder", "Forward", "Forward"],
            CITIZENSHIP: ["USA", "USA", "Canada", "Canada"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 0.0)

    def test_all_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            FIRST_NAME: ["John", "Johnny", "Mike", "Michael"],
            LAST_NAME: ["Doe", "Doee", "Smith", "Smithy"],
            PSEUDONYM: ["JD", "J.D.", "MS", "M.S."],
            HEIGHT: [180, 181, 175, 176],
            DATE_OF_BIRTH: ["1990-01-01", "1990-02-01", "1985-05-20", "1985-06-20"],
            FOOT: ["Right", "Left", "Left", "Right"],
            POSITION: ["Midfielder", "Defender", "Forward", "Midfielder"],
            CITIZENSHIP: ["USA", "Canada", "Canada", "USA"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 1.0)

    def test_partial_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            FIRST_NAME: ["John", "John", "Mike", "Michael"],
            LAST_NAME: ["Doe", "Doe", "Smith", "Smith"],
            PSEUDONYM: ["JD", "JD", "MS", "MS"],
            HEIGHT: [180, 180, 175, 175],
            DATE_OF_BIRTH: ["1990-01-01", "1990-01-01", "1985-05-20", "1985-06-20"],
            FOOT: ["Right", "Right", "Left", "Left"],
            POSITION: ["Midfielder", "Midfielder", "Forward", "Forward"],
            CITIZENSHIP: ["USA", "USA", "Canada", "Canada"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 1/2)



if __name__ == "__main__":
    unittest.main()