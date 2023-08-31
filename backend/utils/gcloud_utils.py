from google.cloud import storage
import hashlib

BUCKET_NAME = 'collage-bucket'
PROJECT_ID = 'hai-gcp-artist-friendly'


def upload_to_bucket(image_stream, blob_name):
    """Uploads a file to a bucket."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(image_stream, content_type='image/png')
    print(f"Uploaded {blob_name} to {BUCKET_NAME}.")
    return blob.public_url

def test_upload(file: str):
    image_stream = open(file, 'rb').read()
    blobname = hashlib.md5(image_stream).hexdigest()
    url = upload_to_bucket(image_stream, blobname)
    return url

if __name__ == '__main__':
    # upload all images in ./assets
    import os
    gallery = []
    for file in os.listdir('./assets'):
        url = test_upload(f'./assets/{file}')
        gallery.append({
            "imgSrc": url,
        })
    import json
    with open('gallery.json', 'w') as f:
        json.dump(gallery, f, indent=4)