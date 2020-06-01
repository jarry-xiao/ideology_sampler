import argparse
import atexit
import random

from IPython.display import display
import psycopg2
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from utils.db import df_to_postgres

estimated_coefs = pd.read_csv("configs/estimated_coefs.csv")

social = {
    "libleft": lambda data: data.social < 0,
    "libright": lambda data: data.social < 0,
    "authleft": lambda data: data.social > 0,
    "authright": lambda data: data.social > 0,
}

economic = {
    "libleft": lambda data: data.economic < 0,
    "libright": lambda data: data.economic > 0,
    "authleft": lambda data: data.economic < 0,
    "authright": lambda data: data.economic > 0,
}


def quadrant_sample(category, max_level=False):
    answers = []
    for i, data in estimated_coefs.iterrows():
        if max_level:
            magnitude = 1.5 
        else:
            magnitude = random.choice([.5, 1.5])
        if data.economic != 0:
            if economic[category](data):
                answers.append(magnitude)
            else:
                answers.append(-magnitude)
        if data.social != 0:
            if social[category](data):
                answers.append(magnitude)
            else:
                answers.append(-magnitude)
    answers = (np.array(answers) + 1.5).astype(int).tolist()
    print(answers)
    return answers


def standard_sampler(mode):
    print(f"Mode: {mode}")
    distributions = {"normal": [0, 1, 2, 3], "extreme": [0, 3]}
    def f(q_id):
        return random.choice(distributions[mode])
    return f


def answer_key(answers):
    def f(q_id):
        return answers[q_id]
    return f


class IdeologySampler:

    def exit_handler(self):
        self.driver.close()

    def __init__(self, N=10000):
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
            if i % 11 == 10:
                print(f"Sample {i}: restarting driver")
                self.restart_driver()
            try:
                modes = ["normal", "extreme"]
                mode = modes[round(random.random())]
                f = standard_sampler(mode)
                self.sample(self.start + i, mode, f)
                i += 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                print(f"Encountered error on trial {i}, retrying...")
                continue

    def sample(self, t_id, mode, f):
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
        display(df)
        success = df_to_postgres(df, "data", self.conn, commit=False)
        if success:
            scores_df = pd.DataFrame(
                [[t_id, economic, social, mode]],
                columns=["t_id", "economic", "social", "mode"]
            )
            display(scores_df)
            df_to_postgres(scores_df, "scores", self.conn)
        if self.load_questions:
            q_df = pd.DataFrame(q_data, columns=["q_id", "q_name", "q_text"])
            df_to_postgres(q_df, "questions", self.conn)

    def quadrant(self, mode, max_level=False):
        i = 0
        while i < self.N:
            # Hack to prevent excessive memory usage by Firefox
            if i % 11 == 10:
                print(f"Sample {i}: restarting driver")
                self.restart_driver()
            try:
                answers = quadrant_sample(mode, max_level=max_level)
                f = answer_key(answers)
                self.sample(self.start + i, mode, f)
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
    parser.add_argument("--mode")
    parser.add_argument("--max_level", action="store_true", default=False)
    args = parser.parse_args()
    if args.N is not None:
        ideology_sampler = IdeologySampler(int(args.N))
    else:
        ideology_sampler = IdeologySampler()
    if args.mode is not None:
        print(args.max_level)
        ideology_sampler.quadrant(args.mode, args.max_level)
    else:
        ideology_sampler()
