from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
# from dataloaders.rawframe_util import RawVideoExtractor

class Video_DataLoader(Dataset):
    def __init__(
            self,
            videofile_path,
            videodata_dir,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0
        ):
        self.video_ids = []
        with open(videofile_path, 'r') as f:
            for line in f.readlines():
                self.video_ids.append(line.strip())

        self.video_paths = []
        for i, video_id in enumerate(self.video_ids):
            video_path = os.path.join(videodata_dir, "{}".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path + '.mp4'
                if os.path.exists(video_path) is False:
                    video_path = video_path.replace(".mp4", ".avi")
                    if os.path.exists(video_path) is False:
                        video_path = video_path.replace(".avi", "")
                        if os.path.exists(video_path) is False:
                            print('video path = {} is not exists.'.format(video_path))
                            break
            self.video_paths.append(video_path)

        self.test_set_start_time = {}
        self.test_set_end_time = {}
        if 'didemo' in videofile_path: # get start and end timestamps
            for k_ in self.video_ids:
                self.test_set_start_time[k_] = 0
                # trick to save time on obtaining each video length
                # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
                # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
                self.test_set_end_time[k_] = 31

        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.video_num = len(self.video_ids)
        print("Video number: {}".format(self.video_num))

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)

    def __len__(self):
        return self.video_num

    def _get_rawvideo(self, choice_video_ids, choice_video_paths):
        # NOTE: numpy==1.24.3时, np.long报错, 改为np.longlong, np.float改为np.float_ 
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        try:
            for i, video_id in enumerate(choice_video_ids):
                video_path = choice_video_paths[i]
                if len(self.test_set_start_time) == 0:
                    raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                else:
                    raw_video_data = self.rawVideoExtractor.get_video_data(video_path, self.test_set_start_time[video_id], self.test_set_end_time[video_id])

                raw_video_data = raw_video_data['video']
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}".format(video_path, video_id))
        except Exception as excep:
            print("video path: {} error. Error: {}".format(video_path, excep))
            pass

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        try:
            video_id = self.video_ids[idx]
            video_path = self.video_paths[idx]
            
            if not os.path.exists(video_path):
                print(f"Warning: Missing video at {video_path}. Skipping...")
                # Trả về tensor rỗng hoặc giá trị mặc định để không làm crash DataLoader
                dummy_video = torch.zeros((self.max_frames, 3, self.resolution, self.resolution))
                dummy_mask = torch.zeros(self.max_frames)
                return video_id, dummy_video, dummy_mask
            
            video, video_mask = self._get_rawvideo([video_id], [video_path])
            return video_id, video, video_mask

        except IndexError:
            print(f"IndexError: idx={idx} is out of range.")
            dummy_video = torch.zeros((self.max_frames, 3, self.resolution, self.resolution))
            dummy_mask = torch.zeros(self.max_frames)
            return None, dummy_video, dummy_mask