import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3, ResNet50V2, NASNetLarge
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries, quickshift

# Load and preprocess images
def load_data(data_path, img_size=(224, 224)):
    images = []
    labels = []
    count = np.zeros(2)  # Assuming two classes: Healthy and Tumor
    
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                img_normalized = img_resized / 255.0
                images.append(img_normalized)
                
                # One-hot encoding for labels
                if label == 'Healthy':
                    labels.append([0, 1])  # Healthy -> [0, 1]
                    count[0] += 1
                else:
                    labels.append([1, 0])  # Tumor -> [1, 0]
                    count[1] += 1
    
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)  # Reshape for CNN input
    labels = np.array(labels)  # Convert labels to numpy array
    
    print(f"Total images processed: {len(images)}")
    print(f"Total healthy images processed: {count[0]}")
    print(f"Total unhealthy images processed: {count[1]}")
    
    return images, labels

# Build model with pre-trained base and custom top layers
def build_model(base_model_class, input_shape=(224, 224, 3), num_classes=2):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='sigmoid')(x)  # Use sigmoid for binary classification
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model

# Create data generator for training and testing
class DataGenerator(Sequence):
    def __init__(self, images, labels, batch_size=16):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        batch_images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Convert grayscale images to RGB on the fly
        batch_images_rgb = np.repeat(batch_images, 3, axis=-1).astype(np.float32)
        
        return batch_images_rgb, batch_labels

# Train the model with the given data generators
def train_model(model, train_generator, test_generator):
    checkpoint_filepath = '/tmp/ckpt/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                                 monitor='val_accuracy',
                                                 mode='max',
                                                 save_best_only=True)

    history = model.fit(train_generator,
                        epochs=10,
                        validation_data=test_generator,
                        callbacks=[model_checkpoint_callback])
    
    return history

# Evaluate the model on test data
def evaluate_model(model, test_generator):
    loss, acc = model.evaluate(test_generator)
    print(f"Test Loss: {loss}, Test Accuracy: {acc}")
    
# Generate LIME heatmap and explanation for a given image
def generate_lime_heatmap_and_explanation(model, image):
    explainer = lime_image.LimeImageExplainer(kernel_width=0.10)
    
    def model_predict_proba(image_array):
        image_array_resized = tf.image.resize(image_array, (224, 224))  # Resize to match model input
        return model.predict(image_array_resized)

    explanation = explainer.explain_instance(
        image,
        model_predict_proba,
        top_labels=2,
        hide_color=1,
        num_samples=1000,
        segmentation_fn=lambda x: slic(x, n_segments=300)
    )
    
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[1],
        positive_only=True,
        hide_rest=False,
        num_features=1000,
        min_weight=0.01
    )
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    heatmap_with_boundaries = mark_boundaries(temp, mask)
    plt.imshow(heatmap_with_boundaries)
    plt.title("Produced Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Explanation Mask")
    plt.axis("off")
    
    plt.show()

# Main function to execute the workflow
def main(data_path):
    # Load data
    images, labels = load_data(data_path)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images,labels,test_size=0.2,stratify=labels,random_state=42)

    # Create data generators
    train_generator = DataGenerator(x_train, y_train)
    test_generator = DataGenerator(x_test, y_test)

    # Build models with different architectures
    models = {
        'InceptionV3': build_model(InceptionV3),
        'ResNet50V2': build_model(ResNet50V2),
        'NASNetLarge': build_model(NASNetLarge)
    }

    # Compile models and train them
    for name in models:
        print(f"Training model: {name}")
        models[name].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        train_model(models[name], train_generator, test_generator)

        # Evaluate the trained model on test data
        evaluate_model(models[name], test_generator)

# Example usage of the main function with your dataset path.
data_path = r"C:\Users\ayush\Desktop\PJT\Brain Tumor Dataset"
main(data_path)