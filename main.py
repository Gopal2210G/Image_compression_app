import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO


def initialize_centroids(X, k):
    centroids = X.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


# Function to assign each data point to the closest centroid
def assign_to_centroids(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


# Function to update centroids based on the mean of assigned data points
def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        if np.sum(labels == i) > 0:  # Check if cluster i has any assigned pixels
            centroids[i, :] = np.mean(X[labels == i, :], axis=0)
        else:
            centroids[i, :] = np.random.uniform(0, 255, X.shape[1])
    return centroids


def compress_image(image, k):
    image_np = np.array(image)
    pixels = image_np.reshape((-1, 3))

    # Initialize centroids randomly
    centroids = initialize_centroids(pixels, k)

    for _ in range(10):
        labels = assign_to_centroids(pixels, centroids)
        centroids = update_centroids(pixels, labels, k)

    # Replace each pixel with its nearest centroid
    compressed_palette = centroids[labels]
    compressed_image = compressed_palette.reshape(image_np.shape).astype('uint8')
    return compressed_image


def main():
    st.title('Image Compression App')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.subheader('Original Image')
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption='Original Image', use_column_width=True)

        k = st.slider('Select number of colors (k)', 2, 256, 16)

        try:
            compressed_image = compress_image(original_image, k)
            st.subheader(f'Compressed Image (k={k})')
            st.image(compressed_image, caption=f'Compressed Image (k={k})', use_column_width=True)
        except Exception as e:
            st.error(f"Error compressing image: {str(e)}")

        compressed_image = Image.fromarray(compressed_image)  # Convert numpy array back to PIL Image
        compressed_img_pil = compressed_image.convert('RGB')  # Ensure image mode is RGB for compatibility
        compressed_img_bytes = BytesIO()  # Create a BytesIO object to hold the image data
        compressed_img_pil.save(compressed_img_bytes, format='JPEG')  # Save PIL Image to BytesIO as JPEG
        compressed_img_bytes.seek(0)  # Reset the BytesIO object's position to the beginning

        st.download_button(
            label="Download Compressed Image",
            data=compressed_img_bytes,
            file_name='compressed_image.jpg',
            mime='image/jpeg'
        )
if __name__ == '__main__':
    main()

