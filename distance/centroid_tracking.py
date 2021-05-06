from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

## Centroid tracking algorithm adapted from:
## https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

class Person():
    def __init__(self, centroid, bbox):
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

    def new_person(self, centroid, bbox):
        person_id = self.next_person

        self.people[person_id] = Person(centroid, bbox)
        self.disappeared[person_id] = 0
        
        self.next_person += 1

    def tracking_update(self, new_bboxes): 
            new_centroids = [self.get_centroids(bbox) for bbox in new_bboxes]

            if len(self.people) == 0:
                for i in range(len(new_centroids)):
                    self.new_person(new_centroids[i], new_bboxes[i])
            else:
                used_rows = set()
                used_cols = set()

                people_ids = list(self.people.keys())
                cur_centroids = [list(self.people.values())[i].centroid for i in range(len(people_ids))]
                distances = dist.cdist(np.array(cur_centroids), np.array(new_centroids))

                rows = distances.min(axis=1).argsort()
                cols = distances.argmin(axis=1)[rows]
                
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    person_id = people_ids[row]
                    self.people[person_id] = Person(new_centroids[col], new_bboxes[col])
                    self.disappeared[person_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

                unused_row = set(range(0, distances.shape[0])).difference(used_rows)
                unused_col = set(range(0, distances.shape[1])).difference(used_cols)

                if distances.shape[0] >= distances.shape[1]:
                    for row in unused_row:
                        person_id = people_ids[row]
                        self.disappeared[person_id] += 1

                else:
                    for col in unused_col:
                        self.new_person(new_centroids[col], new_bboxes[col])
                        
    def tracking_frame(self, new_bboxes):
        if len(new_bboxes) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
            return []
        else:
            # calculate new centroids
            self.tracking_update(new_bboxes)
            people_ids = list(self.people.keys())
            ret_bboxes = []
            for person_id in people_ids:
                if self.disappeared[person_id] == 0:
                    bbox = self.people[person_id].bbox
                    bbox.append(person_id)
                    ret_bboxes.append(bbox)

            return ret_bboxes