import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def sliding_window(data, labels, window_size, step_size):
    X = []
    y = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window_data = data[i:i + window_size]
        window_labels = labels[i:i + window_size]
        # Only keep window if it contains a single class
        if len(np.unique(window_labels)) == 1:
            X.append(window_data)
            y.append(window_labels[0])
    return np.array(X), np.array(y)

def main():
    print('Loading data...')
    df = pd.read_csv('EMG-data.csv')
    
    print('Filtering class 0...')
    df = df[df['class'] != 0].copy()
    
    # We want labels to be 0-indexed for PyTorch (classes are 1-7, so subtract 1)
    df['class'] = df['class'] - 1
    
    print('Extracting windows...')
    window_size = 200
    step_size = 50
    
    X_list = []
    y_list = []
    
    # We can group by label (subject) and find contiguous segments of the same class
    # An easier way is just sliding window over the whole filtered data, but we must ensure
    # we don't mix different subjects or different class segments.
    # The sliding_window function checks if the window contains a single class, which handles class transitions.
    # We should also ensure it doesn't cross subject boundaries.
    
    for subject_id, group in df.groupby('label'):
        data = group[['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']].values
        labels = group['class'].values
        X_subj, y_subj = sliding_window(data, labels, window_size, step_size)
        if len(X_subj) > 0:
            X_list.append(X_subj)
            y_list.append(y_subj)
            
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f'Extracted {len(X)} windows of size {window_size}.')
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print('Normalizing...')
    # Fit scaler on train data
    scaler = StandardScaler()
    # We need to reshape to 2D for scaler, then back to 3D
    num_samples_train, window_size, num_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_channels)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples_train, window_size, num_channels)
    
    num_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, num_channels)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(num_samples_test, window_size, num_channels)
    
    print('Saving data...')
    np.save('X_train.npy', X_train_scaled)
    np.save('X_test.npy', X_test_scaled)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    print('Preprocessing done!')

if __name__ == '__main__':
    main()
