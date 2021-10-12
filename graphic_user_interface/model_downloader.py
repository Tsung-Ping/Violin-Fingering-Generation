import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_pretrained_model():
    model_dir = './pretrained_model.zip'
    if not os.path.exists(model_dir):
        gdd.download_file_from_google_drive(file_id='18cui9MlfrH297ezj9jJoxvRtOJTUmWWY',
                                            dest_path=model_dir,
                                            unzip=True)

if __name__ == "__main__":
    # download pretrained model
    download_pretrained_model()