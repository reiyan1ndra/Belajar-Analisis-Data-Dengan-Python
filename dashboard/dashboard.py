import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import calendar
import streamlit as st
from matplotlib.ticker import ScalarFormatter
from babel.numbers import format_currency
sns.set(style='dark')
from matplotlib.ticker import FuncFormatter
from tabulate import tabulate




def create_daily_orders_df(df):
        daily_orders_df = df.resample(rule='D', on='order_approved_at').agg({
            "order_id": "nunique",
            "payment_value": "sum"
        })
        daily_orders_df = daily_orders_df.reset_index()
        daily_orders_df.rename(columns={
            "order_id": "order_count",
            "payment_value": "revenue"
        }, inplace=True)
        
        return daily_orders_df
    
def create_sum_spend_df(df):
        sum_spend_df = df.resample(rule='D', on='order_approved_at').agg({
            "payment_value": "sum"
        })
        sum_spend_df = sum_spend_df.reset_index()
        sum_spend_df.rename(columns={
            "payment_value": "total_spend"
        }, inplace=True)

        return sum_spend_df

def create_sum_order_items_df(df):
        sum_order_items_df = df.groupby("product_category_name_english")["product_id"].count().reset_index()
        sum_order_items_df.rename(columns={
            "product_id": "product_count"
        }, inplace=True)
        sum_order_items_df = sum_order_items_df.sort_values(by='product_count', ascending=False)

        return sum_order_items_df

def create_shipping_durations_df(df):
    df['shipping_duration'] = df['order_delivered_customer_date'] - df['order_delivered_carrier_date']
    shipping_durations_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        'shipping_duration': 'mean'
    })
    
    shipping_durations_df = shipping_durations_df.reset_index()
    shipping_durations_df.rename(columns={
        'shipping_duration': 'average_shipping_duration'
    }, inplace=True)
    
    return shipping_durations_df

def create_purchase_durations_df(df):
    df['purchase_duration'] = df['order_delivered_customer_date'] - df['order_approved_at']
    
    purchase_durations_df = df.resample(rule='D', on='order_approved_at').agg({
        'purchase_duration': 'mean'
    })
  
    purchase_durations_df = purchase_durations_df.reset_index()
    purchase_durations_df.rename(columns={
        'purchase_duration': 'average_purchase_duration'
    }, inplace=True)
    
    return purchase_durations_df


def review_score_df(df):
        review_scores = df['review_score'].value_counts().sort_values(ascending=True)
        most_common_score = review_scores.idxmax()

        return review_scores, most_common_score


def create_bystate_df(df):
        bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
        bystate_df.rename(columns={
            "customer_id": "customer_count"
        }, inplace=True)
        most_common_state = bystate_df.loc[bystate_df['customer_count'].idxmax(), 'customer_state']
        bystate_df = bystate_df.sort_values(by='customer_count', ascending=False)

        return bystate_df, most_common_state

def create_order_status(df):
        order_status_df = df["order_status"].value_counts().sort_values(ascending=False)
        most_common_status = order_status_df.idxmax()

        return order_status_df, most_common_status

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "payment_value": "sum"
    })
    rfm_df.columns = ["customer_id", "latest_order_date", "frequency", "monetary"]
    rfm_df["latest_order_date"] = pd.to_datetime(rfm_df["latest_order_date"])
    recent_date = df["order_purchase_timestamp"].max()
    rfm_df["recency"] = (recent_date - rfm_df["latest_order_date"]).dt.days
    rfm_df.drop("latest_order_date", axis=1, inplace=True)
    
    return rfm_df


datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_data_df = pd.read_csv("all_data_df.csv")
all_data_df.sort_values(by="order_approved_at", inplace=True)
all_data_df.reset_index(inplace=True)

for column in datetime_cols:
    all_data_df[column] = pd.to_datetime(all_data_df[column])



with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://raw.githubusercontent.com/reiyan1ndra/Belajar-Analisis-Data-Dengan-Python/main/dashboard/olist_ecommerce.png")
    min_date = all_data_df["order_approved_at"].min()
    max_date = all_data_df["order_approved_at"].max()
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_data_df[(all_data_df["order_approved_at"] >= str(start_date)) & 
                (all_data_df["order_approved_at"] <= str(end_date))]

daily_orders_df = create_daily_orders_df(main_df)
sum_spend_df = create_sum_spend_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
review_scores, common_score = review_score_df(main_df)
bystate_df, most_common_state = create_bystate_df(main_df)
shipping_durations_df = create_shipping_durations_df(main_df)
purchase_durations_df = create_purchase_durations_df(main_df)
order_status_df, most_common_status = create_order_status(main_df)
rfm_df = create_rfm_df(main_df)


st.header('Olist E-commerce Dataset :sparkles:')
#Daily Orders
st.subheader('Daily Orders')
col1, col2 = st.columns(2)
 
with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)
 
with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "IDR", locale='id_ID') 
    st.metric("Total Revenue", value=total_revenue)
 
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(
    x=daily_orders_df["order_approved_at"],
    y=daily_orders_df["order_count"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

#Customer Expenditure
st.subheader("Customer Expenditure")
col1, col2 = st.columns(2)
with col1:
    total_spend = format_currency(sum_spend_df["total_spend"].sum(), "IDR", locale="id_ID")
    st.markdown(f"Total Spend: **{total_spend}**")

with col2:
    avg_spend = format_currency(sum_spend_df["total_spend"].mean(), "IDR", locale="id_ID")
    st.markdown(f"Average Spend: **{avg_spend}**")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    sum_spend_df["order_approved_at"],
    sum_spend_df["total_spend"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", labelsize=15)
st.pyplot(fig)


# Ordered Items
st.subheader("Ordered Items")
col1, col2 = st.columns(2)

with col1:
    total_items = sum_order_items_df["product_count"].sum()
    st.markdown(f"Total Items: **{total_items}**")

with col2:
    avg_items = sum_order_items_df["product_count"].mean()
    st.markdown(f"Average Items: **{avg_items}**")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35,15))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="product_count", y="product_category_name_english", hue="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0], legend=False)
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=30)
ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

sns.barplot(x="product_count", y="product_category_name_english", hue="product_category_name_english", data=sum_order_items_df.sort_values(by="product_count", ascending=True).head(5), palette=colors, ax=ax[1], legend=False)
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
st.pyplot(fig)

#Customer Ratings

st.subheader("Review Score")
col1,col2 = st.columns(2)

with col1:
    avg_review_score = review_scores.mean()
    st.markdown(f"Average Review Score: **{avg_review_score}**")

with col2:
    most_common_review_score = review_scores.value_counts().index[0]
    st.markdown(f"Most Common Review Score: **{most_common_review_score}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=review_scores.index, 
            y=review_scores.values, 
            order=review_scores.index,
            palette=["#068DA9" if score == common_score else "#D3D3D3" for score in review_scores.index]
            )

plt.title("Rating by customers for service", fontsize=25)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(fontsize=12)
st.pyplot(fig)

#Purchase Durations
st.subheader("Purchase Durations")
st.caption('Purchase Duration measures the time difference between the initial time when the customers buy the products and received the products')
col1, col2 = st.columns(2)
avg_purchase_duration = purchase_durations_df['average_purchase_duration'].mean()
avg_purchase_duration_days = avg_purchase_duration / np.timedelta64(1, 'D')
with col1:
    st.write(f"Average purchase duration: {avg_purchase_duration_days:.2f} days")

fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(
    x="order_approved_at",
    y="average_purchase_duration",
    data=purchase_durations_df,
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
ax.set_ylabel("Average Purchase Duration (days)", fontsize=20)
ax.set_xlabel("Order Purchase Date", fontsize=20)
ax.invert_yaxis()
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
st.pyplot(fig)

#Shipping Durations
st.subheader("Shipping Durations")
st.caption('Shipping duration measures the time difference between the initial time when the carier picks up the products and the time when customers received the products')
col1, col2 = st.columns(2)
avg_shipping_duration = shipping_durations_df['average_shipping_duration'].mean()
avg_shipping_duration_days = avg_shipping_duration / np.timedelta64(1, 'D')

with col1:
    st.write(f"Average shipping duration: {avg_shipping_duration_days:.2f} days")

fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(
    x="order_purchase_timestamp",
    y="average_shipping_duration",
    data=shipping_durations_df,
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.set_ylabel("Average Shipping Duration (days)", fontsize=15)
ax.set_xlabel("Order Purchase Date", fontsize=15)
ax.invert_yaxis()
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
st.pyplot(fig)

#E-commerce Demographics
st.subheader("E-commerce Demographics")
col1, col2 = st.columns(2)

most_common_state = bystate_df.customer_state.value_counts().index[0]
st.markdown(f"Most Common State: **{most_common_state}**")
fig, ax = plt.subplots(figsize=(20, 10))
 
sns.barplot(x=bystate_df.customer_state.value_counts().index,
                y=bystate_df.customer_count.values, 
                data=bystate_df,
                palette=["#068DA9" if score == most_common_state else "#D3D3D3" for score in bystate_df.customer_state.value_counts().index]
                    )

plt.title("Number customers from State", fontsize=30)
plt.xlabel("State", fontsize=20)
plt.ylabel("Number of Customers", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
st.pyplot(fig)


st.markdown(f"Most Common Order Status: **{most_common_status}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=order_status_df.values,
                y=order_status_df.index,
                order=order_status_df.index,
                palette=["#068DA9" if status == most_common_status else "#D3D3D3" for status in order_status_df.index]
                )
    
plt.title("Order Status", fontsize=20)
plt.xlabel("Status", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
st.pyplot(fig)

#RFM Analysis
st.subheader("Best E-commerce Customer Based on RFM Parameters")
col1, col2, col3 = st.columns(3)

with col1:
     average_recency = round(rfm_df.recency.mean(), 1)
     st.metric("Average Recency (days)", value = average_recency)

with col2:
     average_frequency = round(rfm_df.frequency.mean(), 1)
     st.metric("Average Frequency", value = average_frequency)

with col3:
    average_frequency = format_currency(rfm_df.monetary.mean(), "IDR", locale='id_ID') 
    st.metric("Average Monetary", value = average_frequency)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
color = "#068DA9"

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), color=color, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_xlabel("customer_id", fontsize=30)
ax[0].set_title("By Recency (days)", loc="center", fontsize=30)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)
labels = [label.get_text()[:5] for label in ax[0].get_xticklabels()]
ax[0].set_xticks(range(len(labels)))
ax[0].set_xticklabels(labels)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), color=color, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_xlabel("customer_id", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=30)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)
labels = [label.get_text()[:5] for label in ax[1].get_xticklabels()]
ax[1].set_xticks(range(len(labels)))
ax[1].set_xticklabels(labels)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), color=color, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel(None)
ax[2].set_xlabel("customer_id", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=30)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)
labels = [label.get_text()[:5] for label in ax[2].get_xticklabels()]
ax[2].set_xticks(range(len(labels)))
ax[2].set_xticklabels(labels)

fig.text(0.05, -0.05, 'label: x-axis displays last five digits of customer_id', ha='left', fontsize=20) # Add the short information below the graphs

plt.suptitle("Best Customer Based on RFM Parameters (customer_id)", fontsize=50)

st.pyplot(fig)

st.caption('Copyright (c) Muhammad Reiyan Indra')



