#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# 1. 1D array from 1–20
arr1 = np.arange(1, 21)
print("1D Array:", arr1)

# 2. 2D array of shape 4×5
arr2 = np.arange(1, 21).reshape(4, 5)
print("\n2D Array:\n", arr2)

# 3. 3D array of shape 2×3×3
arr3 = np.arange(1, 19).reshape(2, 3, 3)
print("\n3D Array:\n", arr3)

# 4. Zero matrix 3×3
zero_matrix = np.zeros((3, 3))
print("\nZero Matrix:\n", zero_matrix)

# 5. Matrix of 7s (5×5)
matrix7 = np.full((5, 5), 7)
print("\nMatrix of 7s:\n", matrix7)


# In[2]:


import numpy as np

arr = np.arange(1, 26).reshape(5, 5)

# 1. Extract the 3rd row
third_row = arr[2]
print("3rd row:", third_row)

# 2. Extract the 2nd column
second_column = arr[:, 1]
print("\n2nd column:", second_column)

# 3. 2×2 block from bottom-right corner
block = arr[3:5, 3:5]
print("\n2×2 bottom-right block:\n", block)

# 4. All even numbers
even_numbers = arr[arr % 2 == 0]
print("\nEven numbers:", even_numbers)

# 5. Replace middle element with 999
arr[2, 2] = 999
print("\nArray after replacing middle element:\n", arr)


# In[3]:


import numpy as np

print("===== EXERCISE 3 — RESHAPING =====")

# 1. 1×12 → 3×4
arr1 = np.arange(1, 13).reshape(3, 4)
print("\n1×12 → 3×4:\n", arr1)

# 2. 6×6 → 3×12 → 2×18
arr2 = np.arange(1, 37).reshape(6, 6)

arr_3x12 = arr2.reshape(3, 12)
arr_2x18 = arr2.reshape(2, 18)

print("\n6×6 → 3×12:\n", arr_3x12)
print("\n6×6 → 2×18:\n", arr_2x18)

# 3. Flatten (2D → 1D)
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr3.flatten()
print("\nFlatten 2D → 1D:", flat)

# 4. 1D → Column & Row vectors
arr4 = np.arange(1, 6)
column_vector = arr4.reshape(-1, 1)
row_vector = arr4.reshape(1, -1)

print("\nColumn Vector:\n", column_vector)
print("\nRow Vector:\n", row_vector)


# In[6]:


import numpy as np
arr = np.arange(1, 100001)
squared_loop = []
for x in arr:
    squared_loop.append(x*x)
    squared_vector = arr ** 2
get_ipython().run_line_magic('timeit', 'squared_loop = [x*x for x in arr]')
get_ipython().run_line_magic('timeit', 'squared_vector = arr ** 2')
import time

start = time.time()
squared_loop = [x*x for x in arr]
t1 = time.time() - start

start = time.time()
squared_vector = arr ** 2
t2 = time.time() - start

print("Loop Time:", t1)
print("Vectorization Time:", t2)

if t2 < t1:
    print("➡ Vectorization is faster")
else:
    print("➡ Loop is faster")




# In[8]:


import numpy as np

print("===== EXERCISE 5 — CONDITIONAL SELECTION =====")

# Given marks array
marks = np.array([10, 45, 67, 89, 32, 56, 77, 90, 12])
print("\nOriginal Marks:", marks)

# 1. Extract all passing marks (> 40)
passing = marks[marks > 40]
print("\nPassing Marks (>40):", passing)

# 2. Find marks between 50–80
between = marks[(marks >= 50) & (marks <= 80)]
print("\nMarks between 50–80:", between)

# 3. Replace all marks < 40 with 0
marks[marks < 40] = 0
print("\nAfter replacing marks < 40 with 0:", marks)

# 4. Count how many students passed (> 40)
pass_count = np.sum(marks > 40)
print("\nNumber of students passed:", pass_count)


# In[9]:


import pandas as pd

print("===== EXERCISE 6 — PANDAS SERIES =====")

# Create Series for product names and price list
product_names = pd.Series(["Laptop", "Mouse", "Keyboard", "Monitor", "Printer"])
prices = pd.Series([55000, 450, 1500, 8999, 7800])

print("\nProduct Names:")
print(product_names)

print("\nPrices:")
print(prices)

# 1. Find maximum and minimum price
max_price = prices.max()
min_price = prices.min()

print("\nMaximum Price:", max_price)
print("Minimum Price:", min_price)

# 2. Filter items with price > 500
expensive_items = prices[prices > 500]
print("\nItems With Price > 500:")
print(expensive_items)

# 3. Convert all names to uppercase
uppercase_names = product_names.str.upper()
print("\nProduct Names in Uppercase:")
print(uppercase_names)


# In[10]:


import pandas as pd

print("===== EXERCISE 7 — PANDAS DATAFRAME =====")

# Creating the DataFrame
data = {
    "Name": ["Arun", "Divya", "Rahul", "Sneha", "Kiran", "Meena"],
    "Age": [20, 21, 22, 19, 20, 21],
    "Gender": ["M", "F", "M", "F", "M", "F"],
    "Score": [75, 92, 60, 88, 45, 95]
}

df = pd.DataFrame(data)

print("\nFull DataFrame:")
print(df)

# 1. Display head and tail
print("\nHead (First 5 rows):")
print(df.head())

print("\nTail (Last 5 rows):")
print(df.tail())

# 2. Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# 3. Sort by Score descending
sorted_df = df.sort_values(by="Score", ascending=False)
print("\nSorted by Score (Descending):")
print(sorted_df)

# 4. Filter students with Score > 80
high_scorers = df[df["Score"] > 80]
print("\nStudents with Score > 80:")
print(high_scorers)

# 5. Add new column "Grade" based on Score
def get_grade(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"

df["Grade"] = df["Score"].apply(get_grade)

print("\nDataFrame with Grade column:")
print(df)


# In[11]:


import pandas as pd

print("===== EXERCISE 8 — INDEXING & SLICING =====")

# Given DataFrame
df = pd.DataFrame({
    "Name": ["A", "B", "C", "D", "E"],
    "Age": [20, 21, 22, 23, 24],
    "Score": [88, 76, 90, 66, 95]
})

print("\nOriginal DataFrame:")
print(df)

# 1. Select rows 0 to 2
rows_0_to_2 = df.iloc[0:3]
print("\nRows 0 to 2:")
print(rows_0_to_2)

# 2. Select Score column
score_column = df["Score"]
print("\nScore Column:")
print(score_column)

# 3. Select Age & Score columns
age_score_columns = df[["Age", "Score"]]
print("\nAge & Score Columns:")
print(age_score_columns)

# 4. Select students aged > 21
age_above_21 = df[df["Age"] > 21]
print("\nStudents aged > 21:")
print(age_above_21)

# 5. Select rows where Score > 80
score_above_80 = df[df["Score"] > 80]
print("\nStudents with Score > 80:")
print(score_above_80)


# In[15]:


import pandas as pd
import numpy as np
from io import StringIO

# ---------------- CSV DATA inside code ----------------
csv_data = """Name,Age,Marks,City
ram,20,87,Bangalore
sita,,91,chennai
john,???,abc,mumbai
ram,20,87,Bangalore
meera,21,,Hyderabad
,19,77,Delhi
kiran,22,85,
"""
df = pd.read_csv(StringIO(csv_data))
# -------------------------------------------------------

# 2. Identify missing values
print("Missing values before cleaning:\n", df.isna().sum(), "\n")

# 3 & 4. Age cleaning (convert, replace ???, fill mean)
df["Age"] = pd.to_numeric(df["Age"].replace("???", np.nan), errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 5. Convert Marks to numeric
df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")

# 6. Replace missing City with “Unknown”
df["City"] = df["City"].fillna("Unknown")

# 7. Remove duplicates
df = df.drop_duplicates()

# 8. Convert Name to title case
df["Name"] = df["Name"].str.title()

# 9. Summary statistics
print("\nSummary statistics:\n", df.describe(include="all"), "\n")

# 10. Top 3 highest scorers
top3 = df.sort_values(by="Marks", ascending=False).head(3)
print("Top 3 highest scorers:\n", top3)


# In[16]:


import pandas as pd

# Sample DataFrame
data = {
    "Name": ["Ram", "Sita", "John", "Meera", "Kiran"],
    "Age": [20, 19.5, 21, 22, 23],
    "Marks": [87, 91, 75, 68, 85]
}
df = pd.DataFrame(data)

# 1. Add column “Pass/Fail”  (pass if Marks >= 50)
df["Pass/Fail"] = df["Marks"].apply(lambda m: "Pass" if m >= 50 else "Fail")

# 2. Add Bonus Marks = Marks + 5
df["Bonus Marks"] = df["Marks"] + 5

# 3. Drop unnecessary columns  (example: drop Bonus Marks)
df = df.drop(columns=["Bonus Marks"])

# 4. Rename columns (Marks → Score)
df = df.rename(columns={"Marks": "Score"})

# 5. Change datatype of Age to int
df["Age"] = df["Age"].astype(int)

print(df)


# In[17]:


import pandas as pd

data = {
    "Title": ["Inception","Interstellar","The Dark Knight","Avatar","Titanic","Stranger Things","Money Heist","Breaking Bad","Jawan","RRR","The Conjuring","Squid Game"],
    "Year": [2010,2014,2008,2009,1997,2016,2017,2008,2023,2022,2013,2021],
    "Genre": ["Sci-Fi","Sci-Fi","Action","Fantasy","Romance","Sci-Fi","Thriller","Crime","Action","Action","Horror","Thriller"],
    "Rating": [8.8,8.6,9.0,7.8,7.9,8.7,8.2,9.5,7.2,8.0,7.5,8.1],
    "Duration": [148,169,152,162,195,50,45,47,171,182,112,55],
    "Type": ["Movie","Movie","Movie","Movie","Movie","Series","Series","Series","Movie","Movie","Movie","Series"],
    "Actor": ["Leonardo DiCaprio","Matthew McConaughey","Christian Bale","Sam Worthington","Leonardo DiCaprio","Millie Bobby Brown","Álvaro Morte","Bryan Cranston","Shah Rukh Khan","Ram Charan","Vera Farmiga","Lee Jung-jae"],
    "Country": ["USA","USA","USA","USA","USA","USA","Spain","USA","India","India","USA","South Korea"]
}

df = pd.DataFrame(data)
print(df)


# In[18]:


import pandas as pd

# --- Dataset inside code (Mini Netflix) ---
data = {
    "Title": ["Inception","Interstellar","The Dark Knight","Avatar","Titanic","Stranger Things","Money Heist","Breaking Bad","Jawan","RRR","The Conjuring","Squid Game"],
    "Year": [2010,2014,2008,2009,1997,2016,2017,2008,2023,2022,2013,2021],
    "Genre": ["Sci-Fi","Sci-Fi","Action","Fantasy","Romance","Sci-Fi","Thriller","Crime","Action","Action","Horror","Thriller"],
    "Rating": [8.8,8.6,9.0,7.8,7.9,8.7,8.2,9.5,7.2,8.0,7.5,8.1],
    "Duration": [148,169,152,162,195,50,45,47,171,182,112,55],
    "Type": ["Movie","Movie","Movie","Movie","Movie","Series","Series","Series","Movie","Movie","Movie","Series"],
    "Actor": ["Leonardo DiCaprio","Matthew McConaughey","Christian Bale","Sam Worthington","Leonardo DiCaprio","Millie Bobby Brown","Álvaro Morte","Bryan Cranston","Shah Rukh Khan","Ram Charan","Vera Farmiga","Lee Jung-jae"],
    "Country": ["USA","USA","USA","USA","USA","USA","Spain","USA","India","India","USA","South Korea"]
}
df = pd.DataFrame(data)

# 1. Top-rated movie
top_movie = df[df["Rating"] == df["Rating"].max()]

# 2. Movies after 2015
movies_after_2015 = df[(df["Type"] == "Movie") & (df["Year"] > 2015)]

# 3. All Sci-Fi movies
sci_fi_movies = df[df["Genre"] == "Sci-Fi"]

# 4. Average rating
average_rating = df["Rating"].mean()

# 5. Most common genre
most_common_genre = df["Genre"].mode()[0]

# 6. Longest movie
longest_movie = df[df["Duration"] == df["Duration"].max()]

# 7. Movies grouped by Type
grouped_by_type = df.groupby("Type")["Title"].count()

print("TOP-RATED MOVIE:\n", top_movie, "\n")
print("MOVIES AFTER 2015:\n", movies_after_2015, "\n")
print("ALL SCI-FI MOVIES:\n", sci_fi_movies, "\n")
print("AVERAGE RATING:\n", average_rating, "\n")
print("MOST COMMON GENRE:\n", most_common_genre, "\n")
print("LONGEST MOVIE:\n", longest_movie, "\n")
print("MOVIES GROUPED BY TYPE:\n", grouped_by_type, "\n")


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# PART A – CREATE DATASET (Mini Netflix)
# =====================================================
data = {
    "Title": [
        "Inception", "Interstellar", "The Dark Knight", "Avatar",
        "Titanic", "Stranger Things", "Money Heist", "Breaking Bad",
        "Jawan", "RRR", "The Conjuring", "Squid Game"
    ],
    "Year": [2010, 2014, 2008, 2009, 1997, 2016, 2017, 2008, 2023, 2022, 2013, 2021],
    "Genre": [
        "Sci-Fi", "Sci-Fi", "Action", "Fantasy",
        "Romance", "Sci-Fi", "Thriller", "Crime",
        "Action", "Action", "Horror", "Thriller"
    ],
    "Rating": [8.8, 8.6, 9.0, 7.8, 7.9, 8.7, 8.2, 9.5, 7.2, 8.0, 7.5, 8.1],
    "Duration": [148, 169, 152, 162, 195, 50, 45, 47, 171, 182, 112, 55],  # in minutes
    "Type": [
        "Movie", "Movie", "Movie", "Movie",
        "Movie", "Series", "Series", "Series",
        "Movie", "Movie", "Movie", "Series"
    ],
    "Actor": [
        "Leonardo DiCaprio", "Matthew McConaughey", "Christian Bale", "Sam Worthington",
        "Leonardo DiCaprio", "Millie Bobby Brown", "Álvaro Morte", "Bryan Cranston",
        "Shah Rukh Khan", "Ram Charan", "Vera Farmiga", "Lee Jung-jae"
    ],
    "Country": [
        "USA", "USA", "USA", "USA",
        "USA", "USA", "Spain", "USA",
        "India", "India", "USA", "South Korea"
    ]
}

df = pd.DataFrame(data)
print("===== FULL MINI NETFLIX DATASET =====")
print(df, "\n")

# =====================================================
# PART B – ANALYZE DATASET
# =====================================================

# 1. Top-rated movie/series
top_movie = df[df["Rating"] == df["Rating"].max()]
print("===== TOP-RATED TITLE =====")
print(top_movie, "\n")

# 2. Movies after 2015
movies_after_2015 = df[(df["Type"] == "Movie") & (df["Year"] > 2015)]
print("===== MOVIES AFTER 2015 =====")
print(movies_after_2015, "\n")

# 3. All Sci-Fi movies
sci_fi_movies = df[df["Genre"] == "Sci-Fi"]
print("===== ALL SCI-FI TITLES =====")
print(sci_fi_movies, "\n")

# 4. Average rating
average_rating = df["Rating"].mean()
print("===== AVERAGE RATING OF ALL TITLES =====")
print(average_rating, "\n")

# 5. Most common genre
most_common_genre = df["Genre"].mode()[0]
print("===== MOST COMMON GENRE =====")
print(most_common_genre, "\n")

# 6. Longest movie (by duration)
longest_movie = df[df["Duration"] == df["Duration"].max()]
print("===== LONGEST TITLE =====")
print(longest_movie, "\n")

# 7. Titles grouped by Type (Movie vs Series)
grouped_by_type = df.groupby("Type")["Title"].count()
print("===== COUNT OF TITLES BY TYPE =====")
print(grouped_by_type, "\n")

# =====================================================
# PART C – VISUALIZATIONS
# =====================================================

# 1. Bar plot of average rating by genre
genre_rating = df.groupby("Genre")["Rating"].mean()
plt.figure()
genre_rating.plot(kind="bar")
plt.title("Average IMDb Rating by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# 2. Count of titles (movies + series) per year
plt.figure()
df["Year"].value_counts().sort_index().plot(kind="bar")
plt.title("Count of Titles per Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()

# 3. Histogram of IMDb ratings
plt.figure()
plt.hist(df["Rating"])
plt.title("Histogram of IMDb Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# Find duplicate titles (same movie/series name)
duplicate_titles = df[df["Title"].duplicated(keep=False)]

print("Duplicate titles:\n", duplicate_titles)
print(df["Title"].value_counts()[df["Title"].value_counts() > 1])
df.loc[df["Rating"] < 3, "Rating"] = np.nan
print(df[["Title", "Rating"]])
# Example: convert Duration to string with 'min' (if not already)
df["Duration"] = df["Duration"].astype(str) + " min"
print(df[["Title", "Duration"]].head())
# Extract the number part and convert to int
df["Duration"] = df["Duration"].str.extract(r'(\d+)').astype(int)

print(df[["Title", "Duration"]].head())


# In[ ]:




