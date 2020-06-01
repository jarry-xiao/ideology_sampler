import argparse
import atexit
import random

from IPython.display import display
import psycopg2
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from utils.db import df_to_postgres

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
                self.sample(self.start + i)
                i += 1
            except KeyboardInterrupt:
                break
            except:
                print(f"Encountered error on trial {i}, retrying...")
                continue

    def sample(self, t_id):
        self.reset()
        modes = ["normal", "extreme"]
        mode = modes[round(random.random())]
        distributions = {"normal": [0, 1, 2, 3], "extreme": [0, 3]}
        print(f"Loading data for trial {t_id}, mode: {mode}")
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
                value = random.choice(distributions[mode])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N")
    args = parser.parse_args()
    if args.N is not None:
        ideology_sampler = IdeologySampler(int(args.N))
    else:
        ideology_sampler = IdeologySampler()
    ideology_sampler()
