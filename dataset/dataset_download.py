# Download and unzip dataset
import gdown

url = 'https://drive.google.com/uc?id=1jyvhdZHn0s5Owkr21k5Ff-c96sIQLtEu'
output = 'all_wav.zip'
gdown.download(url, output, quiet=False)
!unzip -q 'all_wav.zip' -d '/content/all_wav'

url = 'https://drive.google.com/uc?id=1vqvn0F0YYhEFbzLgP9wJ36vyInUnO5b5'
output = 'dataset.csv'
gdown.download(url, output, quiet=False)

import gdown
# Download the augmented data from Google Drive
file_urls = {
    'X_augmented.npy': 'https://drive.google.com/uc?id=1-5QQGFBQuL4AO9XTMwb4o7TlTpsoG9M4',
    'y_augmented.npy': 'https://drive.google.com/uc?id=1-7-Oh7Mj2qaVNr8eBAMhvzMubhDoLsSX',
    'input_lengths_augmented.npy': 'https://drive.google.com/uc?id=1-7yhidCIPb2EdKN7ZDweHOf4Q9sNd63a',
    'label_lengths_augmented.npy': 'https://drive.google.com/uc?id=1-77poJQcMc1V5GqnHLazIfjZcvDbvkVi'
}

for file_name, file_url in file_urls.items():
    gdown.download(file_url, f'/content/{file_name}', quiet=False)
