"""PyTorch dataset specification for the fMRI dataset"""

# System
import os

# Externals
import numpy as np
import pandas as pd
import torch

# Locals
# from utils.preprocess import reshape_patch_3d

class FMRIDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the resting-stage fMRI"""

    def __init__(self, data_dir, data_files, target_df=None, time_frames=32, run=0):
        self.data_dir = data_dir
        self.data_files = data_files

        # Which run to use as data for each subject
        self.run = run
        self.time_frames = time_frames

        self.target_df = target_df
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        filepath = os.path.join(self.data_dir, self.data_files[index], os.listdir(self.data_files[index])[self.run])
        data = np.load(filepath)
        img = data['image']

        # Change format (H,W,D,T) -> (T,H,W,D)
        img = img.transpose(3, 0, 1, 2)

        # Getting first T timesteps, in case of inconsistency
        img = img[:self.time_frames,]
        img = torch.from_numpy(img)

        target = None
        if self.target_df:
            target = self.target_df[data['subject']]

        # Apply cropping
        # xcrop, ycrop, zcrop = self.image_crop
        # x = x[xcrop[0] : x.shape[0] - xcrop[1],
        #       ycrop[0] : x.shape[1] - ycrop[1],
        #       zcrop[0] : x.shape[2] - zcrop[1],
        #       0 : self.time_frames]

        # Apply padding
        # x = np.pad(x, self.padding)
        
        # Split into patches (briefly insert dummy batch dim)
        # x = reshape_patch_3d(x[None, :, None], self.patch_size).squeeze(0)

        # Returns img, target (None), if no targets specified

        return img, target

def preprocess_targets(targets_df):
    """ Preprocess target\s dataframe (binarize, etc), depending on the variable """
    # Set subjectkey as index of dataframe
    targets_df = targets_df.set_index('subjectkey')
    # Binarize sex
    targets_df['sex'] = targets_df['sex'].map({1:0, 2:1})

    return targets_df

def get_datasets(data_dir, n_train, n_valid, n_test, **kwargs):

    # Get the list of files
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] # TODO: Shuffle this later

    # Asserting that entire dataset is greater than the total size of train, valid, and test
    assert len(data_files) >= (n_train + n_valid + n_test)
    train_files = data_files[:n_train]
    valid_files = data_files[n_train:n_train+n_valid]
    test_files = data_files[n_train+n_valid:n_train+n_valid+n_test]

    train_targets, valid_targets, test_targets = None, None, None

    # Load target variable (for downstream task)
    if 'target_filepath' in kwargs and 'target_label' in kwargs:
        targets_df = pd.read_csv(kwargs['target_filepath'])
        targets_df = preprocess_targets(targets_df)
        
        # Extract only desired variable from targets_df
        targets_df = targets_df[kwargs['target_label']]

        # Getting train, valid, test subjects
        train_subjects = [s[-15:-11] + '_' + s[-11:] for s in train_files]
        valid_subjects = [s[-15:-11] + '_' + s[-11:] for s in valid_files]
        test_subjects = [s[-15:-11] + '_' + s[-11:] for s in test_files]

        train_targets = targets_df.loc[targets_df.index.isin(train_subjects)]
        valid_targets = targets_df.loc[targets_df.index.isin(valid_subjects)]
        test_targets = targets_df.loc[targets_df.index.isin(test_subjects)]

    train_data = FMRIDataset(data_dir, train_files, target_df=train_targets)
    valid_data = FMRIDataset(data_dir, valid_files, target_df=valid_targets)
    test_data = FMRIDataset(data_dir, test_files, target_df=test_targets)

    # Datasets and loader config
    return train_data, valid_data, test_data, {}

def _test():
    data_dir = '/global/cscratch1/sd/sswwhan/data/abcd-fmriprep-rs-npz/'

    dataset_optional = {
        'target_filepath': '/global/cscratch1/sd/sswwhan/data/demo.total.csv',
        'target_label': 'sex'
    }

    train_data, valid_data, test_data = get_datasets(data_dir, 128, 16, 16, **dataset_optional)

if __name__ == '__main__':
    _test()
