# Coupon Acceptance Prediction

Predicting whether a customer will accept a coupon: LightGBM, 5-fold CV, 77% accuracy, 0.84 ROC-AUC (team hackathon; this is my modeling notebook).

Built during the campus immersion of the Post Graduate Program in Data Science, ML and Generative AI at IIT Roorkee.

> Note: This was a team hackathon. This notebook is my contribution (the modeling pipeline). The team's overall project is at https://github.com/arvindkr568/IITR-Hackthon

## Problem
Predict whether a customer accepts a recommended coupon (Y = 1 or 0) from demographic, behavioral and contextual features.

## Data
- 10,147 training rows and 2,537 test rows
- Target: coupon accepted (56.8% of rows) or not

## Approach
1. Data checks: duplicates, train/test ID overlap, constant and redundant columns
2. Feature engineering: age, income and time converted to numeric values; visit-frequency bands converted to numbers; missing-value flag for the mostly empty car column
3. Model: LightGBM with early stopping
4. Validation: stratified 5-fold cross-validation, with test predictions averaged across the five fold models
5. Threshold check from 0.30 to 0.70 on out-of-fold predictions (the default 0.5 was best)

## Results
| Metric | Value |
|---|---|
| Out-of-fold accuracy | 76.6% (std 0.8% across folds) |
| Out-of-fold ROC-AUC | 0.84 |
| Majority-class baseline | 56.8% |

## Key findings
- Carry-out (74%) and cheap-restaurant (71%) coupons were accepted most; bar coupons least (41%)
- Acceptance was higher at 2 PM (66%) than at 7 AM (50%), and with 1-day expiry (63%) than 2-hour expiry (49%)
- Occupation, coffee-house and bar visit frequency, and income ranked highest in LightGBM feature importance

## Limitations
Early stopping used the validation fold, so the cross-validation score may be slightly optimistic. A separate holdout set would be a stricter test.
