import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews_all

if __name__ == '__main__':

    #mobile_apps_ids = ["com.ubs.swidKXJ.android", "com.csg.cs.dnmb"]
    mobile_apps_ids = ["com.revolut.revolut"]
    files_names = ["revolut-mobile-app-reviews.csv"]
    #files_names = ["ubs-mobile-app-reviews.csv", "credit-suisse-mobile-app-reviews.csv"]
    for index, mobile_app in enumerate(mobile_apps_ids):
        us_reviews = reviews_all(
            mobile_app,
            sleep_milliseconds=0,  # defaults to 0
            lang='en',  # defaults to 'en'
            country='us',  # defaults to 'us'
            sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
        )

        df = pd.DataFrame(np.array(us_reviews), columns=["review"])
        df = df.join(pd.DataFrame(df.pop("review").tolist()))
        df.head()
        df.to_csv(f"../../data/{files_names[index]}")
