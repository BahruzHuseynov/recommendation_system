from fastapi import FastAPI
from pydantic import BaseModel

import random
import re
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

def TDEE(gender, weight, height, age, activity, goal):
    if gender == "female":
        BMR = 655 + 9.6 * weight + 1.8 * height - 4.7 * age
    else:
        BMR = 66 + 13.7 * weight + 5 * height - 6.8 * age
    
    activity_factors = {
        "sedentary" : 1.2,
        "lightly active" : 1.375,
        "moderately active" : 1.55,
        "very active" : 1.725,
        "extremely active" : 1.9
    }
    
    tdee = activity_factors[activity] * BMR
    
    
    if goal == "lose weight":
        tdee *= 0.8 # 20% percent decrease
    elif goal == "muscle gain":
        tdee += 250 # more 250 calories
    
    BMI = weight/height**2
    if BMI > 30: body_type = "obesity"
    elif BMI > 25: body_type = "overweight"
    elif BMI > 18.5: body_type = "normal"
    elif BMI > 17: body_type = "underweight"
    else: body_type = "thinness"
    
    # Actual error is ~10%, but I used 5%
    return tdee * random.uniform(0.95, 1.05), body_type

def meal_time(tdee, consumed_calorie, meal_type):
    meals = {
        "breakfast": [0.22, random.uniform(0.9, 1)],
        "lunch":[0.31, random.uniform(1, 1.1)],
        "dinner":[0.35, random.uniform(0.95, 1.05)]
    }
    meal = meals[meal_type]
    if meal_type == "breakfast":
        return tdee * meal[0] * meal[1]
    
    left_cals = tdee - consumed_calorie
    if left_cals <= 0:
        return -1
    
    if meal_type == "lunch":
        return left_cals * meal[0] * meal[1]
    return left_cals * meal[1]

def fat_calc(tdee_per_meal):
    total_fat = tdee_per_meal * random.uniform(0.2, 0.35)
    saturated_fat = tdee_per_meal * random.uniform(0, 0.1)
    return total_fat / 9, saturated_fat / 9

def category_determination(data, goal, meal_type):
    if meal_type == "breakfast":
        return data[data["RecipeCategory"].isin(["breakfast", "beverages", "fruits", "dessert"])]
    
    ls_w = ["lunch", "beverages", "chicken", "fruits", "special_dietary", "vegetables"]
    other = ["lunch", "beverages", "chicken", "meat", "fish_and_seafood", "international", "others"]
    
    if meal_type == "lunch":
        if goal == "lose weight":
            return data[data["RecipeCategory"].isin(ls_w)]
        else:
            return data[data["RecipeCategory"].isin(other)]
    
    if meal_type == "dinner":
        ls_w = ['dinner'] + ls_w[1:]
        other = ['dinner'] + other[1:]
        if goal == "lose weight":
            return data[data["RecipeCategory"].isin(ls_w)]
        else:
            return data[data["RecipeCategory"].isin(other)]

def cholesterol(meal_type):
    if meal_type == "lunch":
        return random.uniform(0, int(random.uniform(0.2, 0.3) * 200))
    return random.uniform(0, int(random.uniform(0.1, 0.2) * 200))

def sodium(meal_type):
    if meal_type == "lunch":
        return random.uniform(0, int(random.uniform(0.2, 0.3) * 2300))
    return random.uniform(0, int(random.uniform(0.1, 0.2) * 2300))

def carbohydrate(tdee_per_meal, body_type, goal, weight):
    carb_from_cals = tdee_per_meal * random.uniform(0.45, 0.65) / 4
    
    if body_type == "obesity":
        type_carb_diet = 50
    elif body_type == "overweight":
        type_carb_diet = random.randint(100, 150)
    elif body_type == "normal":
        type_carb_diet = tdee_per_meal * random.uniform(0.5, 0.6) / 4
    else:
        type_carb_diet = tdee_per_meal * random.uniform(0.65, 0.8) / 4
    
    if goal == "lose weight":
        goal_carb = random.randint(70, 150)
    elif goal == "muscle gain":
        goal_carb = weight * random.randint(4, 7)
    else:
        return (carb_from_cals + type_carb_diet) / 2
    return (carb_from_cals + type_carb_diet + goal_carb) / 3

def sugar(tdee_per_meal, gender):
    sugar_grams = tdee_per_meal * random.uniform(0, 0.1) / 4
    if gender == "female":
        gender_sugar_grams = 37.5 * random.uniform(0.95, 1.05)
    else:
        gender_sugar_grams = 25 * random.uniform(0.95, 1.05)
    return (sugar_grams + gender_sugar_grams) / 2

def protein(tdee_per_meal, gender, weight, body_type, activity):
    protein_grams = tdee_per_meal * random.uniform(0.1, 0.35) / 4
    if body_type == "obesity" or body_type == "overweight":
        type_protein = 0.5 * weight
    else:
        type_protein = 0.8 * weight
    
    if gender == "female": gender_protein = 46
    else: gender_protein = 56
        
    if activity == "moderately active":
        activity_protein = weight
    elif activity in ["very active", "extremely active"]:
        activity_protein = weight * random.uniform(1.3, 1.6)
    else:
        activity_protein = 0.8 * weight
    
    return (activity_protein + type_protein + gender_protein + protein_grams) / 4

def fiber(meal_type, gender, age):
    if age > 18:
        if gender == "male":
            total = 38
        if gender == "female":
            total = 25
    elif age > 51:
        if gender == "male":
            total = 30
        if gender == "female":
            total = 21
    else:
        total = random.randint(25, 30)
        
    if meal_type == "breakfast":
        return random.uniform(1, 4) * total / 26
    elif meal_type == "lunch":
        return random.uniform(1.5, 8.5) * total / 26
    return random.uniform(4, 13.5) * total / 26


def user_preference_list(data, age, weight, height, gender, goal, activity, consumed_calorie, meal_type):
    tdee, body_type = TDEE(gender, weight, height, age, activity, goal)
    tdee_per_meal = meal_time(tdee, consumed_calorie, meal_type)
    u_fat, u_sat_fat = fat_calc(tdee_per_meal)
    u_carb = carbohydrate(tdee_per_meal, body_type, goal, weight)
    u_sugar = sugar(tdee_per_meal, gender)
    u_protein = protein(tdee_per_meal, gender, weight, body_type, activity)
    u_fiber = fiber(meal_type, gender, age)
    u_cholesterol = cholesterol(meal_type)
    u_sodium = sodium(meal_type)
    new_data = category_determination(data, goal, meal_type)
    
    cols = ['Name', 'RecipeCategory', 'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
            'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'Time_middle', 'Time_short']
    entire_data = new_data[cols]
    X_scaled = entire_data.drop(columns = ["Name", "RecipeCategory"])
    
    knn1 = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn2 = NearestNeighbors(n_neighbors=5)
    knn1.fit(X_scaled)
    knn2.fit(X_scaled)

    user_preferences = {
        'Calories':tdee_per_meal, 'FatContent':u_fat, 'SaturatedFatContent':u_sat_fat,
        'CholesterolContent':u_cholesterol, 'SodiumContent':u_sodium, 
       'CarbohydrateContent':u_carb, 'FiberContent':u_fiber, 'SugarContent':u_sugar,
       'ProteinContent':u_protein, 'Time_middle':0, 'Time_short':1
    }

    user_df = pd.DataFrame(user_preferences, index=[0])
    print(user_preferences)
    return user_df, knn1, knn2, new_data

data = pd.read_csv("cleaned_recipes.csv")

app = FastAPI()

"""
Weight
Height
Age
Gender
Goal: To gain, maintain or lose weight
Lifestyle: Sedentary, Low Active, Medium, Highly Active, Extremely Active
Meal time: Breakfast, Lunch, Dinner
"""

class UserDetail(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    goal: str
    activity: str
    consumed_calorie: float
    meal_type: str

@app.post("/")
def process_user_details(user : UserDetail):
    user_scaled, model1, model2, new_data = user_preference_list(data, user.age, user.weight, user.height, 
                         user.gender, user.goal, user.activity, 
                         user.consumed_calorie, user.meal_type)
    
    distances1, indices1 = model1.kneighbors(user_scaled)
    distances2, indices2 = model2.kneighbors(user_scaled)

    r1 = new_data.sort_values(by="Calories", ascending = False)
    r1 = r1.iloc[indices1[0][:2]]
    r2 = new_data.iloc[indices2[0][:3]]
    combined_data = pd.concat([r2, r1], axis=0, ignore_index=True)

    def clean_it(data):
        data = re.sub(r'c\(|\)|\"', '', data)
        sentences = data.split(', ')
        return " ".join(sentences)

    def send_back(sample, i):
        dct = {
            'Name' : sample["Name"],
            'Category' : sample["RecipeCategory"],
            'Recipe Instructions' : clean_it(sample['RecipeInstructions']),
            'Calories' : sample['Calories']
        }
        return dct

    main_dct = {}
    for i in range(5):
        main_dct[f'Meal {i+1}'] = send_back(combined_data.iloc[i], i + 1)
    
    return main_dct