Real Estate Price Predictor

A simple Python script that uses linear regression to predict house prices based on size, number of bedrooms, and age of the house.

What it does
1. Generates a synthetic dataset of 500 houses with realistic relationships between size, bedrooms, age, and price (plus some random noise).
2. Trains a linear regression model on this data.
3. Asks you to enter details about a house (square footage, bedrooms, age).
4. Predicts the price, along with a 95% confidence range.
5. Shows two charts: how well the model fits the data, and the distribution of prediction errors.

Requirements
Python 3.9 or higher

Usage

Run the script:
```
python3 real-estate-price-predictor.py
```

You'll be prompted to enter:
;House size in square feet (e.g., 2500)
;Number of bedrooms (must be between 2 and 5)
;Age of the house in years (e.g., 10)

The script will then print the predicted price and confidence range, and open a window with two charts showing model accuracy and residuals.

Notes

;The dataset is randomly generated each time using a fixed seed (`np.random.seed(42)`), so results are reproducible.
; Bedroom count is restricted to 2-5 because that's the range the model was trained on; predictions outside this range would be unreliable.
; This is a learning/demo project, not intended for real-world price estimation.


Feel free to use, modify, and share this project.


this is an example interaction of this program:

==================================================
       REAL ESTATE PRICE PREDICTOR ENGINE       
==================================================
Enter House Size (in SqFt, e.g., 2500): 2200
Enter Number of Bedrooms (e.g., 3): 4
Enter Age of the House (in years, e.g., 10): 5
--------------------------------------------------
Base Predicted Market Value: $474,841.12
95% Confidence Range (Accounting for Noise): $415,210.45 to $534,471.79
--------------------------------------------------
