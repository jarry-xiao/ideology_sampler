import argparse
import atexit
import random

from IPython.display import display
import psycopg2
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor

from utils.db import df_to_postgres


class IdeologySampler:

    def exit_handler(self):
        self.driver.close()

    def __init__(self, N=1):
        self.N = N
        self.options = Options()    
        self.options.headless = True
        self.driver = None
        self.restart_driver()
        self.conn = psycopg2.connect(dbname="politicalcompass", user="pc_user")
        self.cur = self.conn.cursor()
        self.cur.execute("select exists(select 1 from questions where 1=1)")
        q_exists = self.cur.fetchone()
        if q_exists and q_exists[0]:
            self.load_questions = False 
        else:
            self.load_questions = True 
        self.cur.execute("select max(t_id) from data")
        d_exists = self.cur.fetchone()
        if d_exists and d_exists[0] >= 0:
            self.start = d_exists[0] + 1 
        else:
            self.start = 0 
        atexit.register(self.exit_handler)

    def restart_driver(self):
        if self.driver is not None:
            self.driver.close()
        self.driver = webdriver.Firefox(options=self.options)
        self.driver.implicitly_wait(5)

    def reset(self):
        self.driver.delete_all_cookies()
        self.driver.get("https://www.politicalcompass.org/test")

    def __call__(self):
        i = 0
        while i < self.N:
            # Hack to prevent excessive memory usage by Firefox
            if i % 10 == 9:
                print(f"Sample {i}: restarting driver")
                self.restart_driver()
            try:
                self.sample(self.start + i, "normal", f=lambda q_id: random.choice([0, 1, 2, 3]))
                i += 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                print(f"Encountered error on trial {i}, retrying...")
                continue

    def sample(self, t_id, mode, f, write=True):
        self.reset()
        print(f"Loading data for trial {t_id}")
        q_data = []
        data = []
        q_id = 0
        for page in range(6):
            print(f"\tPage {page + 1}")
            prompts = (
                self.driver
                .find_elements_by_xpath(".//form")[0]
                .find_elements_by_xpath(".//div/div/div")
            )
            questions = (
                self.driver
                .find_elements_by_xpath(".//form")[0]
                .find_elements_by_xpath(".//legend")
            )
            for i, (prompt, question) in enumerate(zip(prompts, questions)):
                options = prompt.find_elements_by_xpath(".//label")
                q_text = question.text
                value = f(q_id) 
                choice = options[value].find_elements_by_xpath(".//input")[0]
                name = choice.get_attribute("name")
                data.append([t_id, q_id, value])
                if self.load_questions:
                    q_data.append([q_id, name, q_text])
                q_id += 1
                choice.click()
            next_page = self.driver.find_elements_by_xpath(".//button")[0]
            next_page.click()
        results = self.driver.find_elements_by_xpath(".//h2")[2].text.split("\n")
        economic = float(results[0].split(": ")[1])
        social = float(results[1].split(": ")[1])
        print(f"\tSocial: {social}")
        print(f"\tEconomic: {economic}")
        print()
        df = pd.DataFrame(data, columns=["t_id", "q_id", "value"])
        scores_df = pd.DataFrame(
            [[t_id, economic, social, mode]],
            columns=["t_id", "economic", "social", "mode"]
        )
        display(scores_df)
        if write:
            success = df_to_postgres(df, "data", self.conn, commit=False)
            if success:
                df_to_postgres(scores_df, "scores", self.conn)
            if self.load_questions:
                q_df = pd.DataFrame(q_data, columns=["q_id", "q_name", "q_text"])
                df_to_postgres(q_df, "questions", self.conn)


class QuadrantSampler(IdeologySampler):

    social_categories = {
        "libleft": lambda data: data.social < 0,
        "libright": lambda data: data.social < 0,
        "authleft": lambda data: data.social > 0,
        "authright": lambda data: data.social > 0,
    }

    economic_categories = {
        "libleft": lambda data: data.economic < 0,
        "libright": lambda data: data.economic > 0,
        "authleft": lambda data: data.economic < 0,
        "authright": lambda data: data.economic > 0,
    }

    def __init__(
        self,
        N=1,
        econ_p=0.5,
        social_p=0.5,
        e_p_min=0.0,
        e_p_max=1.0,
        s_p_min=0.0,
        s_p_max=1.0,
        mode=None,
    ):
        self.e_p = econ_p
        self.s_p = social_p
        self.e_range = [e_p_min, e_p_max]
        self.s_range = [s_p_min, s_p_max]
        self.mode = mode
        super().__init__(N)
        self.estimated_coefs = self.get_coefs()

    def get_coefs(self):
        data = (
            psql.read_sql("select * from data order by t_id, q_id limit 62000", self.conn)
            .pivot(index="t_id", columns="q_id")
            .value
        )
        data = data.replace({0: -2, 1: -1, 2: 1, 3: 2})
        scores = (
            psql.read_sql("select * from scores order by t_id limit 1000", self.conn)
            .set_index("t_id")
            [["economic", "social"]]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            data, scores, test_size=0.33, random_state=42
        )
        lasso = MultiOutputRegressor(LassoCV(cv=5, random_state=42))
        lasso.fit(X_train, y_train)
        weights = np.array([lasso.estimators_[0].coef_, lasso.estimators_[1].coef_]).T
        coefs = (
            pd.DataFrame(weights, columns=["economic", "social"])
            .reset_index()
            .rename(columns={"index": "q_id"})
        )
        category = (coefs.economic.abs() - coefs.social.abs()).round(2)
        coefs["category"] = None 
        coefs.loc[category >= 0, "category"] = "economic"
        coefs.loc[category < 0, "category"] = "social"
        return coefs

    def __get_p(self, p):
        q = 1 - p
        return p + random.random() * q

    def __get_magnitudes(self, p):
        i = int(random.random())
        return [.5, 1.5][i]

    def generate_answer_key(self, bias):
        answers = []
        # sample a random probability > 0.5
        econ_p = self.__get_p(self.e_p)
        social_p = self.__get_p(self.s_p)
        econ_conviction = random.uniform(*self.e_range)
        social_conviction = random.uniform(*self.s_range)
        print(f"P(opposite economic view for trial | {bias}) = {1 - econ_p}")
        print(f"P(opposite social view for trial | {bias}) = {1 - social_p}")
        print(f"Economic conviction: {econ_conviction}")
        print(f"Social conviction: {social_conviction}")
        for i, data in self.estimated_coefs.iterrows():
            if data.category == "economic":
                magnitude = self.__get_magnitudes(econ_conviction)
                if random.random() > econ_p:
                    magnitude = -magnitude
                if self.economic_categories[bias](data):
                    answers.append(magnitude)
                else:
                    answers.append(-magnitude)
            elif data.category == "social":
                magnitude = self.__get_magnitudes(social_conviction)
                if random.random() > social_p:
                    magnitude = -magnitude
                if self.social_categories[bias](data):
                    answers.append(magnitude)
                else:
                    answers.append(-magnitude)
        answers = (np.array(answers) + 1.5).astype(int).tolist()
        return answers

    def __call__(self, bias=None):
        i = 0
        while i < self.N:
            # Hack to prevent excessive memory usage by Firefox
            if i % 10 == 9:
                print(f"Sample {i}: restarting driver")
                self.restart_driver()
            try:
                if bias is None:
                    curr_bias = ["libleft", "libright", "authright", "authleft"][random.randrange(0, 4)]
                else:
                    curr_bias = bias
                answers = self.generate_answer_key(curr_bias)
                if not self.mode: 
                    mode = f"quadrant(bias={bias}, e_p={self.e_p}, s_p={self.s_p})"
                else:
                    mode = self.mode
                self.sample(self.start + i, mode, f=lambda q_id, a=answers: a[q_id])
                i += 1
            except KeyboardInterrupt:
                break 
            except Exception as e:
                print(e)
                print(f"Encountered error on trial {i}, retrying...")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N")
    parser.add_argument("--bias")
    parser.add_argument("--quadrant", action="store_true", default=False)
    parser.add_argument("--econ_conviction", type=float, default=0.5)
    parser.add_argument("--soc_conviction", type=float, default=0.5)
    parser.add_argument("--e_p_min", type=float, default=0.0)
    parser.add_argument("--e_p_max", type=float, default=1.0)
    parser.add_argument("--s_p_min", type=float, default=0.0)
    parser.add_argument("--s_p_max", type=float, default=1.0)
    parser.add_argument("--mode")
    args = parser.parse_args()
    has_N = args.N is not None
    if not args.quadrant:
        args.bias = None
        sampler = IdeologySampler() if not has_N else IdeologySampler(N=int(args.N))
    else:
        kwargs = {
            "econ_p": args.econ_conviction,
            "e_p_min": args.e_p_min,
            "e_p_max": args.e_p_max,
            "social_p": args.soc_conviction,
            "s_p_min": args.s_p_min,
            "s_p_max": args.s_p_max,
            "mode": args.mode,
        }
        if not has_N:
            sampler = QuadrantSampler(**kwargs)
        else:
            sampler = QuadrantSampler(N=int(args.N), **kwargs)
    if args.bias is not None:
        sampler(bias=args.bias)
    else:
        sampler()
