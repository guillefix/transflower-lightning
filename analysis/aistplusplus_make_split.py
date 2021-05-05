import pandas as pd
import numpy as np

seqs = [x[:-1].split("_") for x in open("analysis/base_filenames.txt", "r").readlines()]

seqs = [{"genre":x[0], "situation":x[1], "camera":x[2], "dancer":x[3], "musicId":x[4], "choreo":x[5]} for x in seqs]

df = pd.DataFrame(seqs)

df["choreo"].unique().size
df["musicId"].unique().size

len(df["choreo"].unique())*len(df["dancer"].unique())

df["dancer"].unique()
[df[df["dancer"]==d]["choreo"] for d in df["dancer"].unique()]

df["musicId"].unique()

songs=[np.random.choice(df[df["genre"]==g]["musicId"],size=1).item() for g in df["genre"].unique()]

song_dancers=sum([[(s,x) for x in np.random.choice(df[df["musicId"]==s]["dancer"],size=2, replace=False).tolist()] for s in songs], [])
# song_dancers

# df[(df["musicId"]=="mBR4") & (df["dancer"]=="d06")]

song_dancer_choreos=sum([[(s,d,x) for x in np.random.choice(df[(df["musicId"]==s) & (df["dancer"]==d)]["choreo"],size=2, replace=False).tolist()] for s,d in song_dancers], [])

len(song_dancer_choreos)

test_data = pd.concat([df[(df["musicId"]==s) & (df["dancer"]==d) & (df["choreo"]==c)].sample(1) for s,d,c in song_dancer_choreos])
# [df[(df["musicId"]==s) & (df["dancer"]==d) & (df["choreo"]==c)] for s,d,c in song_dancer_choreos]

test_data.count()

test_data_seqs = ["_".join([x["genre"], x["situation"], x["camera"], x["dancer"], x["musicId"], x["choreo"]]) for i,x in test_data.iterrows()]

with open("analysis/aistpp_base_filenames_test.txt", "w") as f:
    f.writelines([x+"\n" for x in test_data_seqs])

# train_data = df[~(df["musicId"].isin(test_data["musicId"]))]
train_data = df[~((df["musicId"].isin(test_data["musicId"])) & (df["choreo"].isin(test_data["choreo"])))]
len(train_data)
# song_choreos=[x.tolist() for i,x in test_data[["musicId","choreo"]].iterrows()]
# song_dancer_choreos=[x.tolist() for i,x in test_data[["musicId","dancer","choreo"]].iterrows()]

# count=0
# for i,x in df[["musicId", "dancer", "choreo"]].iterrows():
#     if x.tolist() not in song_dancer_choreos:
#         count+=1
#
# count

# train_data = df[(~df["musicId"].isin(test_data["musicId"])) & (~df["choreo"].isin(test_data["choreo"])))]

train_data.count()
train_data[["musicId","choreo"]].drop_duplicates().count()
train_data[["dancer","choreo"]].drop_duplicates().count()

train_data_seqs = ["_".join([x["genre"], x["situation"], x["camera"], x["dancer"], x["musicId"], x["choreo"]]) for i,x in train_data.iterrows()]

with open("analysis/aistpp_base_filenames_train.txt", "w") as f:
    f.writelines([x+"\n" for x in train_data_seqs])
