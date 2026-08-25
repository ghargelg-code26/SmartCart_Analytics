from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="SmartCart Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "cleaned_customer_shopping_data.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load and lightly standardise the cleaned ETL output."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    string_columns = data.select_dtypes(include="object").columns
    for column in string_columns:
        data[column] = data[column].astype("string").str.strip()

    data["Purchase Date"] = pd.to_datetime(data["Purchase Date"], errors="coerce")
    data["Is_Returned"] = (
        data["Return Status"].str.lower().eq("returned").astype(int)
    )
    data["Purchase_Frequency_Days"] = data["Frequency of Purchases"].map(
        {
            "Weekly": 7,
            "Fortnightly": 14,
            "Monthly": 30,
            "Quarterly": 90,
            "Rarely": 180,
        }
    )
    return data


if not DATA_PATH.exists():
    st.error(
        "The cleaned dataset was not found. Run the ETL notebook first to create "
        f"`{DATA_PATH.name}` in the project folder."
    )
    st.stop()


df = load_data(DATA_PATH)


@st.cache_resource
def train_models(data: pd.DataFrame):
    """Train the return-risk and customer-value prototype models."""
    return_features = [
        "Age", "Gender", "Category", "Brand", "Size", "Color",
        "Discount (%)", "Purchase Amount (₹)",
    ]
    return_frame = pd.get_dummies(data[return_features], drop_first=True)
    return_target = data["Is_Returned"]
    X_train, X_test, y_train, y_test = train_test_split(
        return_frame,
        return_target,
        test_size=0.2,
        random_state=42,
        stratify=return_target,
    )
    return_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    return_model.fit(X_train, y_train)
    return_predictions = return_model.predict(X_test)
    return_probabilities = return_model.predict_proba(X_test)[:, 1]

    customer_profile = (
        data.groupby("Customer ID", as_index=False)
        .agg(
            Total_Spend=("Purchase Amount (₹)", "sum"),
            Previous_Purchases=("Previous Purchases", "max"),
            Frequency=("Frequency of Purchases", "first"),
            Subscription=("Subscription Status", "first"),
            Average_Rating=("Review Rating", "mean"),
        )
    )
    customer_features = pd.get_dummies(
        customer_profile[
            ["Previous_Purchases", "Frequency", "Subscription", "Average_Rating"]
        ],
        drop_first=True,
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        customer_features,
        customer_profile["Total_Spend"],
        test_size=0.2,
        random_state=42,
    )
    value_model = RandomForestRegressor(n_estimators=200, random_state=42)
    value_model.fit(X_train_c, y_train_c)
    value_predictions = value_model.predict(X_test_c)

    metrics = {
        "return_f1": f1_score(y_test, return_predictions, zero_division=0),
        "return_roc_auc": roc_auc_score(y_test, return_probabilities),
        "value_r2": r2_score(y_test_c, value_predictions),
    }
    return (
        return_model,
        return_frame.columns.tolist(),
        customer_features.columns.tolist(),
        value_model,
        metrics,
    )


(
    return_model,
    return_model_columns,
    customer_model_columns,
    value_model,
    model_metrics,
) = train_models(df)


# Custom styling keeps the dashboard clean while preserving Streamlit's controls.
st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    [data-testid="stMetricValue"] { color: #174a7e; }
    .goal-card {
        background: white;
        border-left: 5px solid #2d9cdb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(20, 50, 80, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🛒 SmartCart Analytics")
st.caption(
    "Explore customer behaviour, evaluate the business hypotheses, and identify "
    "opportunities to reduce returns and improve customer value."
)


with st.sidebar:
    st.header("Filters")
    st.caption(f"Source: `{DATA_PATH.name}`")

    def filter_options(column: str) -> list[str]:
        return sorted(df[column].dropna().unique().tolist())

    categories = st.multiselect("Category", filter_options("Category"), default=filter_options("Category"))
    genders = st.multiselect("Gender", filter_options("Gender"), default=filter_options("Gender"))
    channels = st.multiselect(
        "Shopping channel",
        filter_options("Online/Offline"),
        default=filter_options("Online/Offline"),
    )
    subscriptions = st.multiselect(
        "Subscription status",
        filter_options("Subscription Status"),
        default=filter_options("Subscription Status"),
    )
    festivals = st.multiselect(
        "Festival/Sale",
        filter_options("Festival/Sale"),
        default=filter_options("Festival/Sale"),
    )
    discount_range = st.slider(
        "Discount range (%)",
        min_value=int(df["Discount (%)"].min()),
        max_value=int(df["Discount (%)"].max()),
        value=(int(df["Discount (%)"].min()), int(df["Discount (%)"].max())),
    )

    st.divider()
    st.info(
        "Filters apply to the Overview, Returns, Customers, and Discounts tabs. "
        "Use Reset filters in the menu if needed."
    )


filtered = df[
    df["Category"].isin(categories)
    & df["Gender"].isin(genders)
    & df["Online/Offline"].isin(channels)
    & df["Subscription Status"].isin(subscriptions)
    & df["Festival/Sale"].isin(festivals)
    & df["Discount (%)"].between(discount_range[0], discount_range[1])
].copy()


if filtered.empty:
    st.warning("No rows match the selected filters. Please broaden your selection.")
    st.stop()


tab_overview, tab_returns, tab_customers, tab_discounts, tab_predictions, tab_goals = st.tabs(
    [
        "📊 Overview", "🔁 Returns", "👥 Customers", "🏷️ Discounts",
        "🤖 Predictions", "🎯 Goals",
    ]
)


with tab_overview:
    st.subheader("Shopping overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{len(filtered):,}")
    col2.metric("Customers", f"{filtered['Customer ID'].nunique():,}")
    col3.metric("Average purchase", f"₹{filtered['Purchase Amount (₹)'].mean():,.0f}")
    col4.metric("Return rate", f"{filtered['Is_Returned'].mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        category_summary = (
            filtered.groupby("Category", as_index=False)["Purchase Amount (₹)"]
            .mean()
            .sort_values("Purchase Amount (₹)", ascending=False)
        )
        st.plotly_chart(
            px.bar(
                category_summary,
                x="Category",
                y="Purchase Amount (₹)",
                title="Average purchase amount by category",
                color="Category",
            ),
            use_container_width=True,
        )
    with right:
        monthly = (
            filtered.dropna(subset=["Purchase Date"])
            .assign(Month=lambda frame: frame["Purchase Date"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)["Purchase Amount (₹)"]
            .sum()
        )
        st.plotly_chart(
            px.line(
                monthly,
                x="Month",
                y="Purchase Amount (₹)",
                markers=True,
                title="Purchase value over time",
            ),
            use_container_width=True,
        )

    st.subheader("Data preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)


with tab_returns:
    st.subheader("Goal 1: Reduce returns")
    st.write(
        "Identify categories, sizes, colours, and discount levels associated with higher "
        "return risk. The return rate is calculated as the share of transactions marked "
        "`Returned`."
    )

    category_returns = (
        filtered.groupby("Category", as_index=False)
        .agg(Return_Rate=("Is_Returned", "mean"), Transactions=("Customer ID", "size"))
        .sort_values("Return_Rate", ascending=False)
    )
    discount_returns = (
        filtered.groupby("Discount (%)", as_index=False)
        .agg(Return_Rate=("Is_Returned", "mean"), Transactions=("Customer ID", "size"))
        .sort_values("Discount (%)")
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                category_returns,
                x="Category",
                y="Return_Rate",
                color="Return_Rate",
                title="Return rate by category",
                labels={"Return_Rate": "Return rate"},
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.line(
                discount_returns,
                x="Discount (%)",
                y="Return_Rate",
                markers=True,
                title="Return rate by discount percentage",
                labels={"Return_Rate": "Return rate"},
            ),
            use_container_width=True,
        )

    risky = (
        filtered.groupby(["Category", "Size", "Color"], dropna=False)
        .agg(Return_Rate=("Is_Returned", "mean"), Transactions=("Customer ID", "size"))
        .query("Transactions >= 30")
        .sort_values("Return_Rate", ascending=False)
        .head(10)
        .reset_index()
    )
    st.markdown("**Highest-risk product combinations** *(minimum 30 transactions)*")
    st.dataframe(risky, use_container_width=True, hide_index=True)


with tab_customers:
    st.subheader("Goal 2: Understand high-value customers")
    st.write(
        "Compare customer spending, loyalty behaviour, subscription status, and purchase "
        "frequency to support customer segmentation and retention planning."
    )

    customer_profile = (
        filtered.groupby("Customer ID", as_index=False)
        .agg(
            Total_Spend=("Purchase Amount (₹)", "sum"),
            Previous_Purchases=("Previous Purchases", "max"),
            Average_Rating=("Review Rating", "mean"),
            Purchase_Frequency_Days=("Purchase_Frequency_Days", "first"),
            Subscription_Status=("Subscription Status", "first"),
        )
    )
    customer_profile["Customer Tier"] = pd.qcut(
        customer_profile["Total_Spend"],
        q=3,
        labels=["Standard", "Growth", "High value"],
        duplicates="drop",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.box(
                customer_profile,
                x="Subscription_Status",
                y="Total_Spend",
                color="Subscription_Status",
                title="Customer spend by subscription status",
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.scatter(
                customer_profile,
                x="Previous_Purchases",
                y="Total_Spend",
                color="Customer Tier",
                hover_data=["Average_Rating", "Subscription_Status"],
                title="Previous purchases versus total spend",
            ),
            use_container_width=True,
        )

    st.metric("High-value customers", f"{(customer_profile['Customer Tier'] == 'High value').sum():,}")
    st.dataframe(
        customer_profile.sort_values("Total_Spend", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )


with tab_discounts:
    st.subheader("Goal 3: Optimise discounts")
    st.write(
        "Assess whether larger discounts are associated with higher quantities or purchase "
        "values, while monitoring return rates."
    )

    discount_summary = (
        filtered.groupby("Discount (%)", as_index=False)
        .agg(
            Average_Quantity=("Quantity", "mean"),
            Average_Purchase=("Purchase Amount (₹)", "mean"),
            Return_Rate=("Is_Returned", "mean"),
            Transactions=("Customer ID", "size"),
        )
        .sort_values("Discount (%)")
    )
    st.plotly_chart(
        px.scatter(
            discount_summary,
            x="Discount (%)",
            y="Average_Quantity",
            size="Transactions",
            color="Return_Rate",
            hover_data=["Average_Purchase"],
            title="Discount percentage, quantity, and return rate",
            labels={"Average_Quantity": "Average quantity", "Return_Rate": "Return rate"},
        ),
        use_container_width=True,
    )

    festival_summary = (
        filtered.groupby("Festival/Sale", as_index=False)
        .agg(
            Average_Purchase=("Purchase Amount (₹)", "mean"),
            Average_Quantity=("Quantity", "mean"),
            Transactions=("Customer ID", "size"),
        )
        .sort_values("Average_Purchase", ascending=False)
    )
    st.plotly_chart(
        px.bar(
            festival_summary,
            x="Festival/Sale",
            y="Average_Purchase",
            color="Average_Quantity",
            title="Festival and regular-day purchase behaviour",
            labels={"Average_Purchase": "Average purchase amount"},
        ),
        use_container_width=True,
    )


with tab_predictions:
    st.subheader("🤖 Predictive insights")
    st.write(
        "Use the prototype forms below to estimate return risk for an order and "
        "total spend for a customer profile. Predictions support prioritisation; "
        "they do not prove causation."
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Return model F1", f"{model_metrics['return_f1']:.3f}")
    metric2.metric("Return model ROC-AUC", f"{model_metrics['return_roc_auc']:.3f}")
    metric3.metric("Customer-value R²", f"{model_metrics['value_r2']:.3f}")

    st.divider()
    st.subheader("Return-risk prediction")
    return_left, return_right = st.columns(2)
    with return_left:
        prediction_age = st.number_input(
            "Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median()),
        )
        prediction_gender = st.selectbox(
            "Gender", sorted(df["Gender"].dropna().astype(str).unique())
        )
        prediction_category = st.selectbox(
            "Category", sorted(df["Category"].dropna().astype(str).unique())
        )
        prediction_brand = st.selectbox(
            "Brand", sorted(df["Brand"].dropna().astype(str).unique())
        )
    with return_right:
        prediction_size = st.selectbox(
            "Size", sorted(df["Size"].dropna().astype(str).unique())
        )
        prediction_color = st.selectbox(
            "Colour", sorted(df["Color"].dropna().astype(str).unique())
        )
        prediction_discount = st.number_input(
            "Discount (%)",
            min_value=float(df["Discount (%)"].min()),
            max_value=float(df["Discount (%)"].max()),
            value=float(df["Discount (%)"].median()),
        )
        prediction_amount = st.number_input(
            "Purchase Amount (₹)",
            min_value=float(df["Purchase Amount (₹)"].min()),
            max_value=float(df["Purchase Amount (₹)"].max()),
            value=float(df["Purchase Amount (₹)"].median()),
        )

    prediction_frame = pd.DataFrame([{
        "Age": prediction_age,
        "Gender": prediction_gender,
        "Category": prediction_category,
        "Brand": prediction_brand,
        "Size": prediction_size,
        "Color": prediction_color,
        "Discount (%)": prediction_discount,
        "Purchase Amount (₹)": prediction_amount,
    }])
    prediction_frame = pd.get_dummies(prediction_frame, drop_first=True)
    prediction_frame = prediction_frame.reindex(
        columns=return_model_columns, fill_value=0
    )
    return_probability = return_model.predict_proba(prediction_frame)[0, 1]
    if return_probability >= 0.5:
        st.error(f"Estimated return probability: {return_probability:.1%} (high risk)")
    else:
        st.success(f"Estimated return probability: {return_probability:.1%} (low risk)")
    st.progress(float(return_probability))

    st.divider()
    st.subheader("Customer-value prediction")
    value_left, value_right = st.columns(2)
    with value_left:
        value_previous = st.number_input(
            "Previous purchases",
            min_value=int(df["Previous Purchases"].min()),
            max_value=int(df["Previous Purchases"].max()),
            value=int(df["Previous Purchases"].median()),
        )
        value_frequency = st.selectbox(
            "Purchase frequency",
            sorted(df["Frequency of Purchases"].dropna().astype(str).unique()),
        )
    with value_right:
        value_subscription = st.selectbox(
            "Subscription status",
            sorted(df["Subscription Status"].dropna().astype(str).unique()),
        )
        value_rating = st.slider(
            "Average review rating",
            float(df["Review Rating"].min()),
            float(df["Review Rating"].max()),
            float(df["Review Rating"].median()),
            step=0.1,
        )

    value_frame = pd.DataFrame([{
        "Previous_Purchases": value_previous,
        "Frequency": value_frequency,
        "Subscription": value_subscription,
        "Average_Rating": value_rating,
    }])
    value_frame = pd.get_dummies(value_frame, drop_first=True).reindex(
        columns=customer_model_columns, fill_value=0
    )
    estimated_value = value_model.predict(value_frame)[0]
    st.success(f"Estimated total customer value: ₹{estimated_value:,.2f}")


with tab_goals:
    st.subheader("Business goals and recommended actions")
    st.markdown(
        "<div class='goal-card'><b>📉 Goal 1 — Reduce returns</b><br>"
        "Use the Returns tab to identify high-risk category, size, and colour combinations. "
        "Improve size guidance, product descriptions, and quality checks for recurring risk areas.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='goal-card'><b>💎 Goal 2 — Grow customer value</b><br>"
        "Use the Customers tab to identify high-value tiers and compare subscribers with non-subscribers. "
        "Target retention benefits and personalised offers at the right customer tier.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='goal-card'><b>🏷️ Goal 3 — Optimise discounts</b><br>"
        "Use the Discounts tab to balance quantity and purchase value against return rate. "
        "Avoid increasing discounts without checking whether they create profitable demand.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='goal-card'><b>🤖 Predictive prototype</b><br>"
        "Use the Predictions tab to estimate return probability and customer value. "
        "These estimates can help prioritise customer-service interventions and "
        "retention campaigns.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Recommended stakeholder tool")
    st.write(
        "Streamlit with Plotly is the recommended stakeholder interface because it "
        "combines interactive filters, dynamic charts, prediction forms, and a "
        "browser-based experience without requiring stakeholders to run notebooks."
    )

    st.subheader("Data limitations")
    st.info(
        "This dataset records completed transactions, not abandoned carts, so shipping-charge "
        "analysis cannot directly measure cart abandonment. The dashboard shows associations, "
        "not proof of causation."
    )
