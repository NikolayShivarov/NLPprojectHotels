# -*- coding: utf-8 -*-
"""

"""
import pandas as pd

booking_df = pd.read_csv("booking_reviews.csv")
trip_advisor_df = pd.read_csv("tripadvisor_reviews.csv")

#Removing the columns that are not needed

booking_df.drop("reviewed_at", axis=1, inplace=True)
booking_df.drop("reviewed_by", axis=1, inplace=True)
booking_df.drop("images", axis=1, inplace=True)
booking_df.drop("crawled_at", axis=1, inplace=True)
booking_df.drop("url", axis=1, inplace=True)
booking_df.drop("hotel_name", axis=1, inplace=True)
booking_df.drop("hotel_url", axis=1, inplace=True)
booking_df.drop("avg_rating", axis=1, inplace=True)
booking_df.drop("nationality", axis=1, inplace=True)
booking_df.drop("raw_review_text", axis=1, inplace=True)
booking_df.drop("tags", axis=1, inplace=True)
booking_df.drop("meta", axis=1, inplace=True)
booking_df.drop("review_title", axis=1, inplace=True)

booking_df = booking_df[booking_df["review_text"] != "There are no comments available for this review"]

booking_df.rename(columns={"review_text": "Review"}, inplace=True)
booking_df.rename(columns={"rating": "Rating"}, inplace=True)

cols = list(booking_df.columns)
cols = [cols[1]] + [cols[0]]
booking_df = booking_df[cols]

# Save booking_df as CSV
booking_df.to_csv("booking_reviews_processed_numerical.csv", index=False)

# Save trip_advisor_df as CSV
trip_advisor_df.to_csv("tripadvisor_reviews_processed_numerical.csv", index=False)

def convert_booking(rating):
    if rating > 7:
        return "positive"
    else:
        return "negative"

booking_df["Rating"] = booking_df["Rating"].apply(convert_booking)

def convert_tripA(rating):
    if rating > 3:
        return "positive"
    else:
        return "negative"

trip_advisor_df["Rating"] = trip_advisor_df["Rating"].apply(convert_tripA)

# Save booking_df as CSV
booking_df.to_csv("booking_reviews_processed_discrete.csv", index=False)

# Save trip_advisor_df as CSV
trip_advisor_df.to_csv("tripadvisor_reviews_processed_discrete.csv", index=False)
  
    
