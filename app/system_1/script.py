# this script generates real-time fake transaction data
# for more detials, prefer the flow chart.

# import libraries
import random 
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import time

def random_date(start, end):
    """
    function to generate the random date time
    input: start date and end date
    Output: datetime
    """
    # Calculate the total difference in seconds
    delta = end - start
    int_delta = int(delta.total_seconds())
    # Generate a random number of seconds
    random_second = random.randrange(int_delta)
    # Return the new datetime
    return start + timedelta(seconds=random_second)


def generate_data():
    random.seed(a=int(time.time())) # Seed with time for variety
    # transaction type
    type_lst = ["TRANSFER","PAYMENT","CASH_IN","CASH_OUT","DEBIT"]
    type = random.choice(type_lst)
    
    # amount
    # using typical ranges from synthetic datasets
    amount = np.random.uniform(low=1.0, high=100000.0)
    
    # bank
    bank_lst = ["HDFC","ICICI","IDFC","SBI","AXIS","Canara","YES","IndusIND","IDBI"]
    bank = random.choice(bank_lst)
    
    # date time
    d1 = datetime(2024, 1, 1)
    d2 = datetime(2026, 1, 1)
    date_time = random_date(d1,d2)
    
    # transaction ID (12-digit code)
    trans_id = random.randint(a=100000000000, b=999999999999)

    # Balance logic
    oldbalanceOrg = np.random.uniform(low=amount, high=amount * 10)
    
    # Simulate fraud in 5% of cases by causing a balance error
    is_fraud_case = random.random() < 0.05
    
    if type in ["TRANSFER", "CASH_OUT"]:
        if is_fraud_case:
            newbalanceOrig = oldbalanceOrg # balance doesn't decrease
        else:
            newbalanceOrig = oldbalanceOrg - amount
    elif type == "CASH_IN":
        newbalanceOrig = oldbalanceOrg + amount
    else:
        newbalanceOrig = oldbalanceOrg # DEBIT/PAYMENT handles differently usually, but let's keep it simple

    oldbalanceDest = np.random.uniform(low=0.0, high=1000000.0)
    
    if type in ["TRANSFER", "CASH_IN"]:
        if is_fraud_case and type == "TRANSFER":
            newbalanceDest = oldbalanceDest # balance doesn't increase
        else:
            newbalanceDest = oldbalanceDest + amount
    else:
        newbalanceDest = oldbalanceDest

    step = random.randint(a=1, b=743)

    # create the dataframe for input
    # ensure it matches the 7 features expected by the model
    data_input = pd.DataFrame({
        "step": [step],
        "type": [type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    # data to display on cards
    data = {
        "transaction_id" : trans_id,
        "type": type,
        "sender_bank": bank,
        "reciever_bank": random.choice(bank_lst),
        "amount": round(amount, 2),
        "date_time": date_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return data_input, data


if __name__ == "__main__":

    print("generating data...")
    time.sleep(3)
    data_input, data = generate_data()
    
    # print data
    for key,val in data.items():
        print(f"{key} : {val}")