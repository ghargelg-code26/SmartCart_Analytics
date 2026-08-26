# SmartCart_Analytics

SmartCart_Analytics is an end-to-end customer-shopping analysis project. It cleans and explores transaction data, tests business hypotheses, trains machine-learning prototypes, and presents interactive insights through a Streamlit dashboard.

## Project purpose

The project supports three business goals:

1. **Reduce returns** by identifying product and order characteristics associated with return risk.
2. **Grow customer value** by analysing loyalty behaviour and estimating total customer spend.
3. **Optimise discounts** by comparing discount levels with quantity, purchase value, festival activity, and return rates.

The analysis describes associations in completed transactions. It does not establish causation.

## Dataset

The project uses the public customer shopping trends dataset from [Kaggle](https://www.kaggle.com/datasets/rewantbhriguvanshi/customer-shopping-trends-indian/data). The raw file is stored at:

`dataset/customer_shopping_behavior.csv`

The data contains customer demographics, product details, purchase amounts, quantities, discounts, shipping and delivery information, payment methods, ratings, subscription status, purchase frequency, and return status.

## Project structure

```text
.
├── app.py
├── cleaned_customer_shopping_data.csv
├── dataset/
│   └── customer_shopping_behavior.csv
├── jupyter_notebooks/
│   ├── customer_shopping_behaviour_ETL_and_Anlaysis.ipynb
│   ├── ML_PipeLine.ipynb
│   └── EDA_PipeLine.ipynb
├── requirements.txt
├── Procfile
└── setup.sh
```

## Business requirements and hypotheses

### Goal 1 — Reducing returns

Investigate whether return rates vary by category, size, colour, discount, customer characteristics, and purchase amount. The return-risk prototype classifies an order as likely to be returned or not returned.

### Goal 2 — Predicting high-value customers

Group transactions by `Customer ID` and estimate total customer spend using previous purchases, purchase frequency, subscription status, and average review rating. The result can support customer segmentation and retention planning.

### Goal 3 — Optimising discounts

Assess whether higher discounts are associated with increased quantity or purchase value, while monitoring return rates and festival-sale behaviour. Profit-margin data is unavailable, so the analysis cannot calculate true promotional profitability.

The ETL notebook examines ten hypotheses covering:

* Festival or sale activity versus purchase amount, quantity, and rating.
* Discount levels versus quantity and return rate.
* Shipping charge, delivery speed, and purchase amount.
* Delivery time versus online review rating.
* Age and gender versus category preferences.
* Location versus online or offline shopping preference.
* Subscription status versus loyalty measures.
* Review rating versus purchase-frequency interval.
* Payment method versus purchase amount.
* Category, size, and colour combinations versus return rate.

## Project plan and ETL process

The project follows a structured Extract, Transform, Load and Analyse workflow:

1. **Extract:** Load the raw CSV with pandas and inspect its shape, columns, data types, and missing values.
2. **Transform:** Strip whitespace from text fields, convert `Purchase Date` to datetime, replace meaningful missing categories, standardise return and subscription targets, and map purchase-frequency labels to approximate day intervals.
3. **Feature engineering:** Create `Purchase_Year`, `Purchase_Month`, `Purchase_DayOfWeek`, `Is_Returned`, `Is_Subscribed`, and `Purchase_Frequency_Days`.
4. **Load:** Save the cleaned dataset as `cleaned_customer_shopping_data.csv` for reuse by the machine-learning notebook and Streamlit app.
5. **Analyse:** Produce descriptive summaries, correlations, grouped comparisons, hypothesis visualisations, and predictive prototypes.

ETL is completed before EDA because inconsistent whitespace, raw dates, missing values, and mixed categorical labels can create duplicate groups and misleading charts.

## Core statistical concepts (criterion 1.1)

* **Mean:** The arithmetic average, calculated as the total of all values divided by the number of observations. It summarises the typical purchase amount, quantity, or rating but can be affected by extreme values.
* **Median:** The middle value after sorting observations. It is more robust than the mean when transaction values are skewed or contain outliers.
* **Variance and standard deviation:** Variance measures the average squared distance from the mean. Standard deviation is the square root of variance and expresses typical spread in the original unit of measurement.
* **Probability:** Probability represents how likely an event is. In this project, return rate is an empirical probability: the proportion of transactions marked as returned.
* **Hypothesis testing:** A null hypothesis represents no meaningful difference or relationship. A statistical test uses sample evidence to assess whether the observed result would be unlikely under that null hypothesis. A p-value should be considered alongside effect size, confidence intervals, sample size, and business context.

These concepts are foundational because data analysis requires describing central tendency, measuring variation, quantifying uncertainty, and evaluating whether patterns are stronger than random sampling variation.

## Statistical analysis (criterion 1.2)

The notebooks use pandas to calculate group means, medians, correlations, counts, return probabilities, and grouped proportions. Variability can be inspected with variance and standard deviation calculations such as:

```python
df_clean[['Purchase Amount (₹)', 'Quantity', 'Review Rating']].agg(
	['mean', 'median', 'var', 'std']
)
```

The hypothesis analysis compares grouped outcomes and uses correlation as an exploratory measure. The formal testing section now adds Shapiro–Wilk normality diagnostics, Welch's independent t-tests, Mann–Whitney U tests, Kruskal–Wallis tests, chi-squared tests, and Spearman correlations where their assumptions and variable types are appropriate. The section reports the null and alternative hypotheses, test justification, sample sizes, test statistic, raw p-value, Benjamini–Hochberg adjusted p-value, effect size, confidence interval information, plain-language interpretation, and limitations.

Paired t-tests and Wilcoxon signed-rank tests are documented but not applied because this dataset does not contain genuine before-and-after measurements or matched observations. The formal tests use a copy of the cleaned dataframe and do not change the ETL output, existing EDA objects, machine-learning features, or Streamlit inputs. Results remain exploratory and should be confirmed with customer-level methods, confidence intervals for all test types, and additional operational data before decisions are made.

## Machine learning prototypes (criteria 1.3 and 2.2)

### Return-risk classification

The `ML_PipeLine.ipynb` notebook compares Logistic Regression and Random Forest classification. Categorical predictors are one-hot encoded, and the data is split into training and test sets using stratification. Because returned orders are the minority class, balanced class weights are used.

The models are evaluated with:

* **F1 score:** balances precision and recall for the returned-order class.
* **ROC-AUC:** measures how well the model ranks returned and non-returned transactions across thresholds.
* **Classification report:** shows precision, recall, F1 score, and support for both classes.

Logistic Regression provides an interpretable linear baseline. Random Forest was selected as a useful alternative because it can learn non-linear relationships and interactions without requiring feature scaling. The model with the strongest F1 score is treated as the preferred prototype, while its error trade-offs must still be reviewed.

### Customer-value regression

A Random Forest Regressor estimates total customer spend after transactions are aggregated by `Customer ID`. The test-set **R² score** measures the proportion of variation in customer spend explained by the selected predictors. A low or negative R² would indicate that additional features such as recency, transaction count, average order value, channel, and profit contribution are required.

The predictions are prototypes, not guaranteed future outcomes. They should be validated on new, unseen customer data before being used operationally.

## Dashboard and visualisation (criterion 4.1)

`app.py` provides a Streamlit dashboard using pandas, Plotly, and scikit-learn. It includes:

* Sidebar filters for category, gender, shopping channel, subscription status, festival or sale activity, and discount range.
* Overview metrics for transactions, customers, average purchase amount, and return rate.
* Dynamic Plotly charts for category spending, purchase value over time, return rates, customer tiers, discount behaviour, and festival comparisons.
* A Returns tab showing category risk, discount patterns, and high-risk category-size-colour combinations.
* A Customers tab showing subscription comparisons, customer tiers, and total-spend relationships.
* A Discounts tab showing quantity, purchase value, and return-rate patterns.
* A Predictions tab with interactive return-risk and customer-value forms.
* A Goals tab that explains the business narrative, recommended actions, and data limitations.

Streamlit with Plotly was selected as the stakeholder tool because it provides a browser-based interface, interactive controls, responsive charts, and prediction forms without requiring stakeholders to run notebook cells. The notebooks remain useful for transparent analysis and reproducibility, while the app provides a more accessible decision-support experience.

## Storytelling approach

The dashboard guides users from:

1. **What is happening?** — Overview metrics and purchase trends.
2. **Where are risks concentrated?** — Returns by category, discount, size, and colour.
3. **Which customers matter most?** — Customer tiers, subscription comparisons, and loyalty measures.
4. **How should promotions be assessed?** — Quantity and purchase value are shown alongside return rates.
5. **What might happen next?** — Prediction forms estimate return risk and customer value.
6. **What should be remembered?** — The Goals tab explains recommended actions and limitations.

Chart titles, labels, tooltips, metrics, and short explanations are designed to make the findings accessible to both technical and non-technical audiences.

## Limitations and ethical considerations

* The dataset contains completed purchases, not abandoned carts; therefore shipping analysis cannot directly measure cart abandonment.
* The data is synthetic or publicly provided and may not represent all SmartCart customers.
* Observational associations do not prove that discounts, delivery, ratings, or demographics cause outcomes.
* Customer-level predictions may be affected by incomplete history and should not be used to unfairly deny service.
* Demographic attributes should be monitored for bias and used responsibly. Predictions should support review and prioritisation, not automatic exclusion.
* Profit margins, logistics costs, customer recency, and true lifetime-value outcomes are not available.
* Product combinations are filtered to a minimum transaction count to reduce unreliable conclusions from very small groups.

## Learning journey and development roadmap (criterion 4.2)

Key challenges included working with inconsistent categorical values, handling context-dependent missing values, creating reusable paths between notebook folders and the project root, adding interactive widgets, and ensuring that the notebook and Streamlit app used the same encoded model features. These were addressed through explicit cleaning steps, feature engineering, cached model training, aligned one-hot-encoded columns, and repeated syntax and runtime validation.

The project developed practical skills in pandas ETL, exploratory analysis, formal hypothesis testing, hypothesis formulation, Plotly visualisation, Streamlit interaction design, and scikit-learn classification and regression. Future improvements include customer-clustered inference, bootstrap confidence intervals for categorical tests, model cross-validation and calibration, explainability using feature importance or SHAP, better customer-level temporal features, profit-aware discount analysis, automated tests, and deployment monitoring.

Generative AI was used responsibly to support project development by providing suggestions for code structure, formatting, analytical methods, package and module troubleshooting, and visualisation improvements. All suggestions and generated code were critically reviewed, tested, and adapted where necessary before being incorporated into the project. the AI was used for exploration of large suitable ideas which saved a lot of time and effort.

While AI sometimes reduced the need for independent exploration and deep thinking, using it responsibly helped me manage time, overcome technical challenges, and develop my data analysis skills more efficiently.

## Installation and use

Create or activate a Python environment, then install the dependencies from `requirements.txt`. Run the ETL notebook first so that `cleaned_customer_shopping_data.csv` exists. The machine-learning notebook can then be run to inspect model performance and interactive prototypes.

To launch the dashboard, run Streamlit from the project root:

```text
streamlit run app.py
```

## Credits

* Dataset: [Customer Shopping Trends Indian](https://www.kaggle.com/datasets/rewantbhriguvanshi/customer-shopping-trends-indian/data), accessed through Kaggle.
* Visualisation: [Plotly](https://plotly.com/python/) and [Matplotlib](https://matplotlib.org/).
* Dashboard: [Streamlit](https://streamlit.io/).
* Machine learning: [scikit-learn](https://scikit-learn.org/).
* Data processing: [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/).
* code-institute LMS.

**SmartCart_Analytics** is a comprehensive data analysis tool designed to streamline data exploration, analysis, and visualisation. The tool supports multiple data formats and provides an intuitive interface for both novice and expert data scientists.

## Dataset Content

* Data was taken from https://www.kaggle.com/datasets/rewantbhriguvanshi/customer-shopping-trends-indian/data, which is the synthetic data and gets updated accordingly.

## Unfixed Bugs
* None


### Heroku 
### App Information:

* App Name         smart-cart-analytics
* App Link(Domain) https://smart-cart-analytics-0a5e54da0e9e.herokuapp.com/     
* Region           Europe
* Stack            heroku-24
* Frameworks       Python(version 3.12.8)
* GitHub Repo      ghargelg-code26/SmartCart_Analytics
* Heroku Git URL   https://git.heroku.com/smart-cart-analytics.git
* Generation       Cedar


* The App live link is: https://smart-cart-analytics-0a5e54da0e9e.herokuapp.com/ 

* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. From the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App at the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the `.slugignore` file.

## Acknowledgements (optional)

my acknowledgement goes to Vasi Pavaloi & Rory Patrick sheridian from Code Institute for Guidance and support.