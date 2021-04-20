from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Person():
    def __init__(self, bbox, centroid):
        self.bbox = bbox
        self.centroid = centroid

class Centroid_Tracking():
    def __init__(self):
        self.next_person = 0
        self.people = OrderedDict()
        self.disappeared = OrderedDict()

    def get_centroids(self, bbox):
        x_c = (bbox[0] + bbox[2]) / 2
        y_c = (bbox[1] + bbox[3]) / 2
        return [x_c, y_c]

    def new_person(self, centroid):
        person_id = self.next_person

        self.people[person_id] = centroid
        self.disappeared[person_id] = 0
        
        self.next_person += 1

    def tracking_update(self, new_bboxes): 
            new_centroids = [get_centroids[bbox] for bbox in new_bboxes]

            used_rows = set()
            used_cols = set()

            people_ids = list(self.people.keys())
            cur_centroids = list(self.people.values())

            distances = dist.cdist(np.array(cur_centroids), np.array(new_centroids))

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                person_id = people_ids[row]
                self.people[person_id] = new_centroids[col]
                self.disappeared[person_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_row = set(range(0, distances.shape[0])).difference(used_rows)
            unused_col = set(range(0, distances.shape[1])).difference(used_cols)

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    person_id = people_ids[row]
                    self.disappeared[person_id] += 1

            else:
                for col in unused_cols:
                    self.new_person(new_centroids[col])
                        
    def tracking_frame(self, new_bboxes):
        if len(new_bboxes) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
            return []
        else:
            # calculate new centroids
            self.tracking_update(new_bboxes)

            return []