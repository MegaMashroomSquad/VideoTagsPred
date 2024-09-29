import numpy as np

def iou_metric(ground_truth, predictions):
    iou = len(set.intersection(set(ground_truth), set(predictions)))
    iou = iou / (len(set(ground_truth).union(set(predictions))))
    return iou

def split_tags(tag_list):
    final_tag_list = []
    for tag in tag_list:
        tags = tag.split(": ")
        if len(tags) == 3:
            final_tag_list.append(tags[0])
            final_tag_list.append(tags[0] + ": " + tags[1])
            final_tag_list.append(tags[0] + ": " + tags[1] + ": " + tags[2])
        elif len(tags) == 2:
            final_tag_list.append(tags[0])
            final_tag_list.append(tags[0] + ": " + tags[1])
        elif len(tags) == 1:
            final_tag_list.append(tags[0])
        else:
            print("NOT IMPLEMENTED!!!!", tag)
    return final_tag_list

def find_iou_for_sample_submission(df, pred_df):
    df["tags_new"] = df["tags"].apply(lambda l: l.split(', '))
    df["tags_split"] = df["tags_new"].apply(lambda l: split_tags(l))

    pred_df["predicted_tags_split"] = pred_df["predicted_tags"].apply(lambda l: split_tags(l))
    
    iou_total = 0
    counter = 0
    for i, row in df.iterrows():
        predicted_tags = pred_df[pred_df["video_id"] == row["video_id"]]["predicted_tags_split"].values[0]
        iou_temp = iou_metric(row['tags_split'], predicted_tags)
        iou_total += iou_temp
        counter += 1

    return iou_total / counter