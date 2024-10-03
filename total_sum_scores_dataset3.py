import pandas as pd

df = pd.read_csv('dataset3.csv')

def obtain_total_sum():
    # Skip the first column of a row and obtain rest of data in row
    answer_columns = df.columns[1:] 

    # List to store sum values for each participant
    participant_sums = []

    # Calculate sum for each answer by participant (row) and append to the list
    for answer, row in df.iterrows():
    
        # Skip the first column (participant ID) and sum the rest of the answers
        participant_sum = row[answer_columns].sum()  
        participant_sums.append(participant_sum)

    # Add the total sum for all answers to the participant_sums variable
    total_sum_all_participants = sum(participant_sums)
    participant_sums.append(total_sum_all_participants)
    print(participant_sums)

obtain_total_sum()
