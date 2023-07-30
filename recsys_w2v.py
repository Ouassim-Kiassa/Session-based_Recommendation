import pandas as pd
from gensim.models import Word2Vec
import csv

## I. Load the data
train_sessions = pd.read_csv("~/shared/data/project/training/train_sessions.csv")
item_features = pd.read_csv("~/shared/data/project/training/item_features.csv")
train_purchases = pd.read_csv("~/shared/data/project/training/train_purchases.csv")

test_sessions = pd.read_csv("~/shared/data/project/test/test_sessions.csv")

## II. Process the data

## II.A. Train data
# combine the session and purchase data
combined_sessions = pd.concat([train_sessions, train_purchases])

# double weight on purchases
combined_sessions = pd.concat([combined_sessions, train_purchases])

# sort the data by session_id and by date
combined_sessions = combined_sessions.sort_values(by=['session_id', 'date']).reset_index(drop=True)

## II.B. Test data
test_sessions = test_sessions.sort_values(by=['session_id','date']).reset_index(drop=True)

## III. Creat W2V dataset

# create a dataset that w2v can train/test on
def create_w2v_data(df, type):
    if type == 'train':
        sessions = []
    else:
        sessions = {}
    session = []
    for index, value in df.iterrows():
        if index != 0:
            if str(value["session_id"]) == str(df.at[index-1, "session_id"]):
                session.append(str(value["item_id"]))
            else:
                if len(session) != 0:
                    if type == 'train':
                        sessions.append(session)
                    else:
                        sessions[value["session_id"]] = session
                session = [str(value["item_id"])]
        else:
            session.append(str(value["item_id"]))
    return sessions

# create helper function that check which session was the longest (needed for training the model)
def check_data(list_data):
    lengths = 0
    longest = 0

    for session in list_data:
        lengths += len(session)
        if len(session) > longest:
            longest = len(session)
    avg_nr_items = lengths/len(list_data)
    print("Avg number of items per session:", avg_nr_items, "\nMost viewed items within one single session:", longest)
    return longest

# run helper functions for training data
train_sessions_list = create_w2v_data(combined_sessions, type='train') # this will take a few minutes
longest = check_data(train_sessions_list)

# run helper function for test data
test_sessions_dict = create_w2v_data(test_sessions, type='test')
test_sessions_dict = dict(sorted(test_sessions_dict.items())) # sort the output dict by keys

## IV. Training the model

model = Word2Vec(sentences=train_sessions_list, vector_size=100, window=longest, min_count=2, workers=4)
model.save("word2vec_recsys.model")
#model = Word2Vec.load("word2vec_recsys.model")

## V. Creating the results file
header = ['session_id', 'item_id', 'rank']

with open('results_w2v.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
    
    for session in test_sessions_dict.keys():
        prediction = model.predict_output_word(context_words_list=test_sessions_dict[session], topn=100)
        if prediction != None:
            for rank, pred in enumerate(prediction):
                data = [session, pred[0], rank+1]
                # write the data
                writer.writerow(data)
