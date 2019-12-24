from unittest import TestCase
import pandas as pd

file = "./data/conversion_data.csv"
data = pd.read_csv(file)
df = pd.DataFrame(data)

class TestGet_categorical_variables(TestCase):
    def test_get_categorical_variables(self):
        from build import get_categorical_variables
        res = get_categorical_variables(df=df)
        self.assertTrue("country" in res)
        self.assertTrue("new_user" in res)
        self.assertTrue("source" in res)
        self.assertTrue("converted" in res)

    def test_get_numerical_variables(self):
        from build import get_numerical_variables
        res = get_numerical_variables(df=df)
        self.assertTrue("age" in res)
        self.assertTrue("total_pages_visited" in res)

    def test_get_numerical_variables_percentile(self):
        from build import get_numerical_variables_percentile
        res = get_numerical_variables_percentile(df=df)
        self.assertTrue(isinstance(res, pd.DataFrame))

    def test_get_categorical_variables_modes(self):
        from build import get_categorical_variables_modes
        res = get_categorical_variables_modes(df=df)
        self.assertTrue(isinstance(res, pd.DataFrame))

    def test_get_missing_values_count(self):
        from build import get_missing_values_count
        res = get_missing_values_count(df=df)
        self.assertTrue(isinstance(res, pd.DataFrame))