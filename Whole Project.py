import os
import warnings 
warnings.filterwarnings('ignore')

from glob import glob
import numpy as np 
import pandas as pd
    
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


meta = pd.read_csv("/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")


base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

# Dictionary to map image IDs to their corresponding file paths

imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))
}

meta['path'] = meta['image_id'].map(imageid_path_dict.get)


import plotly.express as px
import plotly.io as pio

# Calculate the value counts for each diagnosis
dx_counts = meta['dx'].value_counts()

# Choose a color palette for the bars (you can choose any other palette from px.colors)
color_palette = px.colors.qualitative.Pastel

# Create a bar chart using plotly express
fig = px.bar(x=dx_counts.values, y=dx_counts.index, orientation='h', color=dx_counts.index,
             color_discrete_sequence=color_palette)

# Customize the layout
fig.update_layout(title_text='Diagnosis Count',
                  xaxis_title='Count',
                  yaxis_title='Diagnosis',
                  xaxis_tickangle=-45,
                  plot_bgcolor='rgba(0, 0, 0, 0)',      # Transparent background
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
                  font=dict(color='rgb(64, 64, 64)')   # Dark gray font color
                  )

# Display the plot in a new window
pio.show(fig)


meta['dx_code'] = pd.Categorical(meta['dx']).codes

import plotly.express as px

# Create a histogram using plotly express
fig = px.histogram(meta, x="age", nbins=20, color="dx", color_discrete_sequence=px.colors.qualitative.Plotly,
                   title="Age Distribution")

# Customize the layout
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
                  font=dict(color='rgb(64, 64, 64)'),  # Dark gray font color
                  xaxis_title='Age',
                  yaxis_title='Count')

# Show the plot
fig.show()


import plotly.express as px

# Calculate the value counts for each sex
sex_counts = meta['sex'].value_counts()

# Choose a color palette for the bars (you can choose any other palette from px.colors)
color_palette = px.colors.qualitative.Pastel

# Create a bar chart using plotly express
fig = px.bar(x=sex_counts.values, y=sex_counts.index, orientation='h', color=sex_counts.index,
             color_discrete_sequence=color_palette)

# Customize the layout
fig.update_layout(title_text='Sex Count',
                  xaxis_title='Count',
                  yaxis_title='Sex',
                  xaxis_tickangle=-45,
                  plot_bgcolor='rgba(0, 0, 0, 0)',      # Transparent background
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
                  font=dict(color='rgb(64, 64, 64)')   # Dark gray font color
                  )

# Show the plot
fig.show()


import plotly.express as px

# Calculate the value counts for each localization
localization_counts = meta['localization'].value_counts()

# Choose a color palette for the bars (you can choose any other palette from px.colors)
color_palette = px.colors.qualitative.Pastel

# Create a bar chart using plotly express
fig = px.bar(x=localization_counts.values, y=localization_counts.index, orientation='h', color=localization_counts.index,
             color_discrete_sequence=color_palette)

# Customize the layout
fig.update_layout(title_text='Localization Count',
                  xaxis_title='Count',
                  yaxis_title='Localization',
                  xaxis_tickangle=-45,
                  plot_bgcolor='rgba(0, 0, 0, 0)',      # Transparent background
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
                  font=dict(color='rgb(64, 64, 64)')   # Dark gray font color
                  )

# Show the plot
fig.show()


print(meta.isnull().sum())


# Replace null with mean
meta['age'].fillna((meta['age'].mean()), inplace=True)


import plotly.graph_objects as go

# Calculate the value counts for each dx type
value_counts = meta['dx_type'].value_counts()

# Define custom colors for the bars
bar_colors = ['rgb(255, 127, 80)', 'rgb(144, 238, 144)', 'rgb(135, 206, 250)', 'rgb(255, 215, 0)']

# Create a bar chart using plotly
fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, marker=dict(color=bar_colors))])

# Customize the layout
fig.update_layout(title='Distribution of 4 different classes of dx type',
                  xaxis_title='dx Type',
                  yaxis_title='Count',
                  xaxis_tickangle=-45,
                  plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',  # Light gray background
                  font=dict(color='rgb(64, 64, 64)')  # Dark gray font color
                  )

# Show the plot
fig.show()


import plotly.graph_objects as go

# Calculate the value counts for each localization
value_counts = meta['localization'].value_counts()

# Define custom colors for the bars
bar_colors = ['rgb(255, 127, 80)', 'rgb(144, 238, 144)', 'rgb(135, 206, 250)', 'rgb(255, 215, 0)']

# Create a bar chart using plotly
fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, marker=dict(color=bar_colors))])

# Customize the layout
fig.update_layout(title='Distribution of classes of localization',
                  xaxis_title='dx Type',
                  yaxis_title='Count',
                  xaxis_tickangle=-45,
                  plot_bgcolor='rgba(0, 0, 0, 0)', 
                  paper_bgcolor='rgba(240, 240, 240, 0.8)',
                  font=dict(color='rgb(64, 64, 64)')
                  )

# Show the plot
fig.show()

from tqdm import tqdm
import numpy as np
from PIL import Image

image_paths = list(meta['path'])
meta['image'] = [np.asarray(Image.open(path).resize((100, 75)), dtype=np.float32) / 255.0 for path in tqdm(image_paths)]

# Create a list of image samples
image_samples = np.random.choice(meta.shape[0], 7 * 5, replace=False)

# Create a figure and subplots
fig, axes = plt.subplots(7, 5, figsize=(4 * 5, 3 * 7))

# Plot the image samples
for i, ax in enumerate(axes.flatten()):
    row = meta.iloc[image_samples[i]]
    ax.imshow(row['image'])
    ax.set_title(row['dx_code'])
    ax.axis('off')

# Save the figure
fig.savefig('category_samples.png', dpi=300)


from keras.utils import to_categorical

X = meta['image']
y = to_categorical(meta['dx_code'])


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


print(X_train.shape , "\n")
print(X_train[1].shape , "\n")


# Convert Pandas Series to NumPy arrays
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

# Reshape image data in 3 dimensions (height = 75, width = 100, channel = 3)
X_train = X_train.reshape(X_train.shape[0], 75, 100, 3)
X_test = X_test.reshape(X_test.shape[0], 75, 100, 3)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# Function to plot the confusion matrix with Plotly
def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        colorscale = 'Plasma'
    else:
        colorscale = 'Magma_r'  # Reversed 'Magma' colorscale for non-normalized matrix

    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(
                {
                    'x': classes[j],
                    'y': classes[i],
                    'text': str(cm[i, j]),
                    'showarrow': False,
                    'font': {'color': 'red' if cm[i, j] > 0.5 else 'black'}
                }
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(classes),
        y=list(classes),
        colorscale=colorscale,
        colorbar=dict(title='Normalized' if normalize else 'Count'),
        showscale=True,
        hoverinfo='z'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label'),
        annotations=annotations
    )

    if normalize:
        fig.update_layout(title_text='Normalized Confusion Matrix')
    else:
        fig.update_layout(title_text='Confusion Matrix (Counts)')

    fig.show()



    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the pre-trained DenseNet-121 model (weights pre-trained on ImageNet)
base_model = DenseNet121(weights='imagenet', include_top=False)

# Freeze some layers in the base model
num_layers_to_freeze = 95  # Choose the number of layers you want to freeze
for layer in base_model.layers[:num_layers_to_freeze]:
    layer.trainable = False

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)  # Add BatchNormalization layer for better convergence
predictions = Dense(7, activation='softmax')(x)

# Create the final model
model_dense = Model(inputs=base_model.input, outputs=predictions)

# Learning Rate Scheduler
initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,   # Adjust decay_steps 
    decay_rate=0.9       # Adjust decay_rate 
)
optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model_dense.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Fit the model
epochs = 60
batch_size = 60
history = model_dense.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                    epochs=epochs, verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                                    callbacks=[learning_rate_reduction,early_stopping], validation_data=(X_test, y_test))

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report

classes = range(7)
    
# Y_true (true labels) and Y_pred_classes (predicted labels)
Y_pred = model_dense.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix with the new colorscale
plot_confusion_matrix(confusion_mtx, classes=classes, normalize=False)

report = classification_report(Y_true, Y_pred_classes)
print(f"Classification Report for <<DenseNet121>> : ")
print(report)


split_proportion = 0.8

# Randomly shuffle the rows of the DataFrame
shuffled_data = meta.sample(frac=1, random_state=41)  # random_state for reproducibility

# Calculate the number of rows for the first piece
total_rows = shuffled_data.shape[0]
split_size = int(total_rows * split_proportion)

# Split the DataFrame into two pieces
Train = shuffled_data.iloc[:split_size]
Test = shuffled_data.iloc[split_size:]

from keras.utils import to_categorical

X_train_image = Train['image']
y_train = to_categorical(Train['dx_code'])

X_test_image = Test['image']
y_test = to_categorical(Test['dx_code'])


# Convert Pandas Series to NumPy arrays
X_train_image = np.array(X_train_image.tolist())
X_test_image = np.array(X_test_image.tolist())

# Reshape image data in 3 dimensions (height = 75, width = 100, channel = 3)
X_train_image = X_train_image.reshape(X_train_image.shape[0], 75, 100, 3)
X_test_image = X_test_image.reshape(X_test_image.shape[0], 75, 100, 3)


from sklearn.preprocessing import StandardScaler

categorical_data_train = Train[[ 'age' ,'dx_type' ,"localization" ]]
categorical_data_test = Test[[ 'age' ,'dx_type' ,"localization" ]]

# Define the columns you want to scale
columns_to_scale = ['age']

# Create a StandardScaler object
scaler = StandardScaler()

# Reshape the column to a 2D array before fitting the scaler
categorical_data_train[columns_to_scale] = scaler.fit_transform(categorical_data_train[columns_to_scale].values.reshape(-1, 1))
categorical_data_test[columns_to_scale] = scaler.transform(categorical_data_test[columns_to_scale].values.reshape(-1, 1))

# List of columns to one-hot encode
columns_to_encode = ['dx_type', 'localization']

# Perform one-hot encoding using get_dummies
encoded_train = pd.get_dummies(categorical_data_train, columns=columns_to_encode)
encoded_test = pd.get_dummies(categorical_data_test, columns=columns_to_encode)

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


categorical_input = Input(shape=(20,))

# Load the pre-trained DenseNet-121 model (Same as before)
base_model = DenseNet121(weights='imagenet', include_top=False)

# Freeze some layers in the base model
num_layers_to_freeze = 95
for layer in base_model.layers[:num_layers_to_freeze]:
    layer.trainable = False

# After GlobalAveragePooling2D layer, concatenate with the categorical input
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

# Add Custom Layers for Categorical Data
y = Dense(64, activation='relu')(categorical_input)
x = Concatenate()([x, y])

x = BatchNormalization()(x)
predictions = Dense(7, activation='softmax')(x)

# Create the final model with both image and categorical inputs
model_dense = Model(inputs=[base_model.input, categorical_input], outputs=predictions)

# Learning Rate Scheduler
initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model_dense.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


# Train the model with data augmentation and callbacks
epochs = 60
batch_size = 60
history = model_dense.fit(
    [X_train_image, encoded_train],  # Update with your actual image and categorical training data
    y_train, epochs=epochs, batch_size=batch_size,
    callbacks=[learning_rate_reduction , early_stopping], validation_data=([X_test_image, encoded_test], y_test)
)

# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

from sklearn.metrics import confusion_matrix, classification_report

Y_pred = model_dense.predict([X_test_image, encoded_test])
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix with the new colorscale
plot_confusion_matrix(confusion_mtx, classes=classes, normalize=False)

report = classification_report(Y_true, Y_pred_classes)
print(f"Classification Report for <<DenseNet121>> : ")
print(report)


import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

categorical_input = Input(shape=(20,))

# Load the pre-trained ResNet-101 model 
base_model = ResNet101(weights='imagenet', include_top=False)

# Freeze some layers in the base model
num_layers_to_freeze = 300
for layer in base_model.layers[:num_layers_to_freeze]:
    layer.trainable = True

# After GlobalAveragePooling2D layer, concatenate with the categorical input
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

# Add Custom Layers for Categorical Data
y = Dense(64, activation='relu')(categorical_input)
x = Concatenate()([x, y])

x = BatchNormalization()(x)
predictions = Dense(7, activation='softmax')(x)

# Create the final model with both image and categorical inputs
model_resnet = Model(inputs=[base_model.input, categorical_input], outputs=predictions)

# Learning Rate Scheduler
initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)

model_resnet.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model with data augmentation and callbacks
epochs = 60
batch_size = 60
history = model_resnet.fit(
    [X_train_image, encoded_train],
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[learning_rate_reduction ,early_stopping ],
    validation_data=([X_test_image, encoded_test], y_test)
)

# Plot the training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report

Y_pred = model_resnet.predict([X_test_image, encoded_test])
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix with the new colorscale
plot_confusion_matrix(confusion_mtx, classes=classes, normalize=False)

report = classification_report(Y_true, Y_pred_classes)
print(f"Classification Report for <<DenseNet121>> : ")
print(report)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.keras.models import Model
from keras.optimizers import Adam

# Create base InceptionV3 model 
base_model = InceptionV3(input_shape=(75, 100, 3), 
                         include_top=False, weights='imagenet')

# Freeze the base model 
base_model.trainable = False

# Add pooling and new output layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) 
predictions = Dense(7, activation='softmax')(x)

# Create new model 
model_inception = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model_inception.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


# Fit the model
epochs = 60
batch_size = 32
history = model_inception.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs,verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction,early_stopping],validation_data = (X_test,y_test))

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()


# Y_true (true labels) and Y_pred_classes (predicted labels)
Y_pred = model_inception.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix with the new colorscale
plot_confusion_matrix(confusion_mtx, classes=classes, normalize=False)

report = classification_report(Y_true, Y_pred_classes)
print(f"Classification Report for <<InceptionV3>> : ")
print(report)